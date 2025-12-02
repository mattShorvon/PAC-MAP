import os
import sys
from pathlib import Path
from spn.io.file import from_file
from spn.actions.learn import Learn
from spn.learn import gens
from spn.data.partitioned_data import PartitionedData
from spn.actions.map_algorithms.max_product import (
    max_product_with_evidence_and_marginals,
)
from spn.actions.map_algorithms.argmax_product import (
    argmax_product_with_evidence_and_marginalized,
)
from spn.actions.map_algorithms.max_search import max_search, forward_checking
from spn.actions.map_algorithms.pac_map import pac_map
from spn.actions.map_algorithms.pac_map_hammingdist import pac_map_hamming
from experiment_scripts.lbp import lbp
from spn.utils.graph import full_binarization
from spn.utils.evidence import Evidence
import argparse
import pandas as pd
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
parser.add_argument('-q', '--q-percent', type=float,
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float,
                    help="Proportion of evidence variables")
parser.add_argument('-m', '--methods', nargs='+', required=True,
                    help='MAP algos to run, separated by space (e.g. MP AMP)')
parser.add_argument('--learn', action='store_true',
                    help='Learn SPN from scratch')
parser.add_argument('--no-learn', dest='learn', action='store_false',
                    help='Use existing SPN')
parser.add_argument('--file-mode', choices=['.map','.query/.evid'], 
                    default='.map',
                    help='Format of query/evidence files')
parser.add_argument('--no-res-file', action="store_true",
                    help="No single file to write the results to, will write" \
                    "to many individual files in each dataset's subfolder instead")
parser.add_argument('--results-file', default='benchmark_results.csv',
                    help='Path to file to store results in, if storing in ' \
                    'a single results file')
parser.add_argument('-dt', '--date',
                    default=datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                    help='Date and time of experiment')

parser.set_defaults(learn=False)

args = parser.parse_args()
use_all = args.all_datasets
data_path = args.data_path
if use_all:
    datasets = sorted(
        [folder.name for folder in os.scandir(data_path) if folder.is_dir()]
    )
else:
    datasets = args.datasets
if args.q_percent and args.e_percent:
    q_percent = args.q_percent
    e_percent = args.e_percent
else: 
    q_percent, e_percent = None, None
methods = args.methods
learn_spn = args.learn
query_evid_filemode = args.file_mode
no_results_file = args.no_res_file
results_filename = args.results_file
datetime_str = args.date

print(f"Datasets: {datasets}")
print(f"MAP methods being run: {methods}")
print(f"Learning SPN from scatch: {learn_spn}")

# Run the experiment
for dataset in datasets:
    # Set up the SPN
    print(f"Running benchmark on dataset {dataset}")
    if learn_spn:
        print("Learning SPN ...")
        kclusters = 10
        pval = 0.1
        try:
            # Try the first pattern
            data = PartitionedData(
                Path(f"{data_path}/{dataset}/{dataset}-train.data"), 1.0
            )
        except:
            # If that fails, try the second pattern
            try:
                data = PartitionedData(
                    Path(f"{data_path}/{dataset}/{dataset}.train.data"), 1.0
                )
            except Exception as error:
                print(f"Failed to load training data: {error}")
                print(f"Error type: {type(error).__name__}")
                run_success = False
                continue 
            
        spn = gens(data.scope, data.training_data, kclusters, pval)
        spn.root = True
        print("SPN learned:", spn.vars(), "vars and", spn.arcs(), "arcs")
    else:
        try:
            spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
            print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")
        except FileNotFoundError as error:
            print(".spn file doesn't exist in this subfolder")
            print(error)
            continue

    # Set up the evidences and queries
    queries, evidences = [], []
    if query_evid_filemode == ".query/.evid":
        with open(f"{data_path}/{dataset}/{dataset}.query") as f:
            for line in f:
                query = [spn.scope()[int(var_id)] for var_id in line.split()[1:]]
                queries.append(query)
        with open(f"{data_path}/{dataset}/{dataset}.evid") as f:
            for line in f:
                evid_info = line.split()[1:]
                index = 0
                evidence = Evidence()
                while index < len(evid_info):
                    var_id = int(evid_info[index])
                    val = int(evid_info[index + 1])
                    evidence[spn.scope()[var_id]] = [val]
                    index += 2
                evidences.append(evidence)
    elif query_evid_filemode == ".map":
        # If the proportion of query and evidence variables has been specified
        # for this experiment, load the files with them in the name
        if q_percent and e_percent:
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
        # Otherwise, load the original .map files that didn't have them in the
        # names
        else:
            with open(f"{data_path}/{dataset}/{dataset}.map") as f:
                for line_no, line in enumerate(f):
                    if line_no % 2 == 0: 
                        query = [spn.scope()[int(var_id)] for var_id in line.split()[1:]]
                        queries.append(query)
                    else: 
                        evid_info = line.split()[1:]
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
        p_evid = spn.value(e)
        print("Evidence    :", e)
        print("Query       :", ' '.join([f"{v.id}({v.n_categories})" for v in q ]))
        print("Marginalized:", ' '.join([f"{v.id}({v.n_categories})" for v in m ]))
        print()
        run_success = True
        if "MP" in methods:
            start = time.perf_counter()
            mp_est_full, _ = max_product_with_evidence_and_marginals(
                spn, e, m
            )
            mp_est = Evidence({var: vals for var, vals in mp_est_full.items() 
                if var not in m})
            mp_prob = spn.value(mp_est) / p_evid
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
            amp_prob = spn.value(amp_est) / p_evid
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
                    time_limit=60, # just giving it 1 minute for this run-through, don't care that much about MS's results anyway
                    marginalized_variables=m,
                    evidence=e,
                )
                ms_prob = spn.value(ms_est) / p_evid
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
                hbp_prob = spn_bin.value(hbp_est) / p_evid
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
                spn, e, m
            )
            # pac_map_prob = spn.value(pac_map_est)
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
                spn, e, m, eta = 0.9, batch_size=1 # These are debug params
            )
            # pac_map_prob = spn.value(pac_map_est)
            pac_map_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "PAC_MAP_Hamming",
                "MAP Estimate": str({var.id: pac_map_est[var] for var in q}),
                "MAP Probability": pac_map_prob,
                "Runtime": pac_map_time
            })
            print(f"PAC MAP Hamming:           {pac_map_prob:.4g}")
            print("MAP Est:", ' '.join([str(pac_map_est[v]) for v in q]))
            print()
    if run_success:
        results_dt = pd.DataFrame(results)
        if q_percent and e_percent:
            results_dt['Query Proportion'] = q_percent
            results_dt['Evid Proportion'] = e_percent
        if no_results_file is False:
            file_exists = os.path.isfile(results_filename)
            results_dt.to_csv(results_filename, mode='a', header=not file_exists, index=False)
        else:
            results_dt.to_csv(f"{data_path}/{dataset}/{dataset}_results.csv", index=False)
