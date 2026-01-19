import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from spn.io.file import from_file
from spn.actions.learn import Learn
from spn.learn import gens
from spn.data.partitioned_data import PartitionedData
from spn.actions.map_algorithms.independent_map import independent_map
from spn.actions.map_algorithms.max_product import (
    max_product_with_evidence_and_marginals,
)
from spn.actions.map_algorithms.argmax_product import (
    argmax_product_with_evidence_and_marginalized,
)
from spn.actions.map_algorithms.max_search import max_search, forward_checking
from spn.actions.map_algorithms.pac_map import pac_map
from spn.actions.map_algorithms.pac_map_hammingdist import pac_map_hamming
from spn.actions.map_algorithms.pac_map_topk import pac_map_topk
from experiment_scripts.lbp import lbp
from spn.utils.graph import full_binarization
from spn.utils.evidence import Evidence
import argparse
from datetime import datetime
import time

# Check command line arguments and initialise variables
# USAGE: python benchmark.py -d <names of dataset folders separated by space 
# e.g. iris nltcs> -m <names of MAP algos to run, separated by space> 
# --no-learn --file-mode .map --data-path test --results-file results.csv
# EXAMPLE: python benchmark.py -m MP AMP MS --no-learn --data-path 20-datasets 
# -d tmovie tretail voting -q 0.4 -e 0.4
parser = argparse.ArgumentParser(description='MAP Benchmark Experiment Params')
parser.add_argument('-all','--all-datasets', action='store_true',
                    help="Iterate through all dataset folders" \
                    "(containing queries, evidences and spns) in the specified"\
                    "data path, or just use specific ones?")
parser.add_argument('-dp', '--data-path', default='test_inputs', 
                    help='Path to folder with all subfolders of spns, ' \
                    'queries and evidence in them')
parser.add_argument('-d', '--datasets', nargs='+',
                    help='Dataset names separated by space (e.g., iris nltcs)')
parser.add_argument('-q', '--q-percent', type=float, required=True,
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float, required=True,
                    help="Proportion of evidence variables")
parser.add_argument('-m', '--methods', nargs='+', required=True,
                    help='MAP algos to run, separated by space (e.g. MP AMP)')
parser.add_argument('--no-res-file', action="store_true",
                    help="No single file to write the results to, will write" \
                    "to many individual files in each dataset's subfolder instead")
parser.add_argument('--results-file', default='benchmark_results.csv',
                    help='Path to file to store results in, if storing in ' \
                    'a single results file')
parser.add_argument('-dt', '--date',
                    default=datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                    help='Date and time of experiment')
parser.add_argument('-id', '--experiment-id', default=1,
                    help='If running several experiments that you want to be ' \
                    'paired together, assign them the same id')

args = parser.parse_args()
use_all = args.all_datasets
data_path = args.data_path
if use_all:
    datasets = sorted(
        [folder.name for folder in os.scandir(data_path) if folder.is_dir()]
    )
else:
    datasets = args.datasets
q_percent = args.q_percent
e_percent = args.e_percent
methods = args.methods
no_results_file = args.no_res_file
results_filename = args.results_file
datetime_str = args.date
experiment_id = args.experiment_id
try:
    n_jobs = int(os.environ.get('SLURM_NTASKS'))
    print(f"On cluster, n_jobs set to {n_jobs}")
except TypeError:
    print("Not on cluster, n_jobs set to -2")
    n_jobs = -2 

print(f"Datasets: {datasets}")
print(f"MAP methods being run: {methods}")

# Run the experiment
for dataset in datasets:
    # Set up the SPN
    print(f"Running benchmark on dataset {dataset}")
    spn_path = Path(f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e.spn")
    try:
        spn = from_file(spn_path)
        print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")
    except FileNotFoundError as error:
        print(".spn file doesn't exist in this subfolder")
        print(error)
        continue

    # Set up the evidences and queries
    queries, evidences = [], []
    with open(
        f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e.map"
        ) as f:
        for line_no, line in enumerate(f):
            if line_no % 2 == 0: 
                query = [spn.scope()[int(var_id)] for var_id in line.split()]
                queries.append(query)
            else: 
                evid_info = line.split()
                index = 0
                evidence = Evidence()
                while index < len(evid_info):
                    var_id = int(evid_info[index])
                    val = int(evid_info[index + 1])
                    evidence[spn.scope()[var_id]] = [val]
                    index += 2
                evidences.append(evidence)
    
    # Loop through each evidence and query combo
    results = []
    for q, e in zip(queries, evidences):
        m = [var for var in spn.scope() if var not in q and var not in e]
        mp_est, amp_est, ms_est, hbp_est = ["None"] * 4
        mp_prob, amp_prob, ms_prob, hbp_prob = [0] * 4
        p_evid = spn.log_value(e)
        print("Evidence    :", e)
        print("Query       :", ' '.join([f"{v.id}({v.n_categories})" for v in q ]))
        print("Marginalized:", ' '.join([f"{v.id}({v.n_categories})" for v in m ]))
        print()
        run_success = True
        if "IND" in methods:
            start = time.perf_counter()
            baseline_est, baseline_prob = independent_map(
                spn, spn_path, e, p_evid, m, n_jobs=n_jobs
            )
            baseline_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "Independent",
                "MAP Estimate": str({var.id: baseline_est[var] for var in q}),
                "MAP Probability": baseline_prob,
                "Runtime": baseline_time
            })
            print(f"Baseline:           {baseline_prob:.4g}")
            print("Baseline Est:", ' '.join([str(baseline_est[v]) for v in q]))
            print()
        if "MP" in methods:
            start = time.perf_counter()
            mp_est_full, _ = max_product_with_evidence_and_marginals(
                spn, e, m
            )
            mp_est = Evidence({var: vals for var, vals in mp_est_full.items() 
                if var not in m})
            mp_prob = spn.log_value(mp_est) - p_evid
            mp_prob = np.exp(mp_prob)
            mp_time = time.perf_counter() - start
            
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "Max Product",
                "MAP Estimate": str({var.id: mp_est[var] for var in q}),
                "MAP Probability": mp_prob,
                "Runtime": mp_time
            })
            print(f"MP:           {mp_prob:.4g}")
            print("MAP Est:", ' '.join([str(mp_est[v]) for v in q]))
            print()
        if "AMP" in methods:
            start = time.perf_counter()
            amp_est, _ = argmax_product_with_evidence_and_marginalized(
                spn, e, m
            )
            filtered_sample = Evidence({var: vals for var, vals in amp_est.items() 
                                            if var not in m})
            amp_prob = spn.log_value(amp_est) - p_evid
            amp_prob = np.exp(amp_prob)
            amp_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "ArgMax Product",
                "MAP Estimate": str({var.id: amp_est[var] for var in q}),
                "MAP Probability": amp_prob,
                "Runtime": amp_time
            })
            print(f"AMP:           {amp_prob:.4g}")
            print("MAP Est:", ' '.join([str(amp_est[v]) for v in q]))
            print()
        if "MS" in methods:
            try:
                start = time.perf_counter()
                ms_est, _ = max_search(
                    spn,
                    forward_checking,
                    time_limit=60,
                    marginalized_variables=m,
                    evidence=e,
                )
                ms_prob = spn.log_value(ms_est) - p_evid
                ms_prob = np.exp(ms_prob)
                ms_time = time.perf_counter() - start
                results.append({
                    "Date": datetime_str,
                    "Dataset": dataset,
                    "Query": str([query.id for query in q]),
                    "Method": "Max Search",
                    "MAP Estimate": str({var.id: ms_est[var] for var in q}),
                    "MAP Probability": ms_prob,
                    "Runtime": ms_time
                })
                print(f"MS:           {ms_prob:.4g}")
                print("MAP Est:", ' '.join([str(ms_est[v]) for v in q]))
                print()
            except TimeoutError as error:
                print("MS timed out")
                print(error)
        if "HBP" in methods:
            try:
                spn_bin = full_binarization(spn)
                spn_bin.fix_scope()
                spn_bin.fix_topological_order()
                start = time.perf_counter()
                hbp_est_full = lbp(spn_bin, e, m, num_iterations=5)
                hbp_est = Evidence({var: vals for var, vals in hbp_est_full.items() 
                                if var not in m})
                hbp_prob = spn_bin.log_value(hbp_est) - p_evid
                hbp_prob = np.exp(hbp_prob)
                hbp_time = time.perf_counter() - start
                results.append({
                    "Date": datetime_str,
                    "Dataset": dataset,
                    "Query": str([query.id for query in q]),
                    "Method": "Hybrid Belief-Propagation",
                    "MAP Estimate": str({var.id: hbp_est[var] for var in q}),
                    "MAP Probability": hbp_prob,
                    "Runtime": hbp_time
                })
                print(f"HBP:           {hbp_prob:.4g}")
                print("MAP Est:", ' '.join([str(hbp_est[v]) for v in q]))
                print()
            except Exception as error:
                print(f"HBP failed with error {error}")
                print(f"Error type: {type(error).__name__}")
                run_success = False
                break 
        if "PACMAP" in methods:
            start = time.perf_counter()
            pac_map_est, pac_map_prob = pac_map(
                spn, spn_path, e, m, batch_size=5000, err_tol=0.01, fail_prob=0.01,
                sample_cap=100000, n_jobs=n_jobs
            )
            # pac_map_prob = spn.log_value(pac_map_est)
            pac_map_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "PAC_MAP",
                "MAP Estimate": str({var.id: pac_map_est[var] for var in q}),
                "MAP Probability": pac_map_prob,
                "Runtime": pac_map_time
            })
            print(f"PAC MAP:           {pac_map_prob:.4g}")
            print("MAP Est:", ' '.join([str(pac_map_est[v]) for v in q]))
            print()
        if "PACMAP-H" in methods:
            start = time.perf_counter()
            pac_map_est, pac_map_prob = pac_map_hamming(
                spn, spn_path, e, m, batch_size=5000, err_tol=0.01, fail_prob=0.01,
                sample_cap=100000, n_jobs=n_jobs
            )
            # pac_map_prob = spn.log_value(pac_map_est)
            pac_map_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": f"PAC_MAP_Hamming",
                "MAP Estimate": str({var.id: pac_map_est[var] for var in q}),
                "MAP Probability": pac_map_prob,
                "Runtime": pac_map_time
            })
            print(f"PAC MAP Hamming:           {pac_map_prob:.4g}")
            print("MAP Est:", ' '.join([str(pac_map_est[v]) for v in q]))
            print()
        if "PACMAP-TopK" in methods:
            start = time.perf_counter()
            pac_map_est, pac_map_prob = pac_map_topk(
                spn, e, m, k=100, batch_size=100, err_tol=0.01, fail_prob=0.01
            )
            # pac_map_prob = spn.log_value(pac_map_est)
            pac_map_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "PACMAP-TopK",
                "MAP Estimate": str({var.id: pac_map_est[var] for var in q}),
                "MAP Probability": pac_map_prob,
                "Runtime": pac_map_time
            })
            print(f"PAC MAP TopK:           {pac_map_prob:.4g}")
            print("MAP Est:", ' '.join([str(pac_map_est[v]) for v in q]))
            print()
    if run_success:
        results_dt = pd.DataFrame(results)
        results_dt['Query Proportion'] = q_percent
        results_dt['Evid Proportion'] = e_percent
        if experiment_id:
            results_dt['Experiment ID'] = experiment_id
        if no_results_file is False:
            file_exists = os.path.isfile(results_filename)
            results_dt.to_csv(results_filename, mode='a', header=not file_exists, index=False)
        else:
            results_dt.to_csv(f"{data_path}/{dataset}/{dataset}_results.csv", index=False)
