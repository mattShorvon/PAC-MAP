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
from spn.actions.map_algorithms.merlin import merlin, make_uai_file, markov_code
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
    spn_path = Path(f"{data_path}/{dataset}/{dataset}.spn")
    try:
        spn = from_file(spn_path)
        print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")
    except FileNotFoundError as error:
        print(".spn file doesn't exist in this subfolder")
        print(error)
        continue

    # Convert the spn to a pgm
    make_uai_file(spn, f"{data_path}/{dataset}/{dataset}.uai")
    print("Converted spn to pgm")

    # Set up the evidences and queries
    queries, evidences = [], []
    with open(
        f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e_nomargs.query"
    ) as f:
        for line in f:
            queries.append(line.strip('\n'))
    with open(
        f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e_nomargs.evid"
    ) as f:
        for line in f:
            evidences.append(line.strip('\n'))

    # Global time budget (seconds) per dataset
    time_budget = 120.0
    dataset_start = time.perf_counter()

    # Loop through each evidence and query combo
    results = []
    print("Starting benchmark")
    run_success = True

    for i, (q, e) in enumerate(zip(queries, evidences), start=1):
        # Check remaining time before starting this query
        elapsed_dataset = time.perf_counter() - dataset_start
        remaining_dataset = time_budget - elapsed_dataset
        if remaining_dataset <= 0:
            # No time left: add zero-probability entries for all remaining queries
            for j in range(i, len(queries) + 1):
                q_line = queries[j - 1]
                q_var_indices = [int(var) for var in q_line.split()[1:]]
                if "WMB" in methods:
                    results.append({
                        "Date": datetime_str,
                        "Dataset": dataset,
                        "Query": q_var_indices,
                        "Method": "WMB Elimination",
                        "MAP Estimate": "NA",
                        "MAP Probability": 0.0,
                        "Runtime": 0.0,
                        "Query Proportion": q_percent,
                        "Evid Proportion": e_percent,
                        "Experiment ID": experiment_id,
                        "Query_ID": j
                    })
                if "AAOBF" in methods:
                    results.append({
                        "Date": datetime_str,
                        "Dataset": dataset,
                        "Query": q_var_indices,
                        "Method": "AAOBF",
                        "MAP Estimate": "NA",
                        "MAP Probability": 0.0,
                        "Runtime": 0.0,
                        "Query Proportion": q_percent,
                        "Evid Proportion": e_percent,
                        "Experiment ID": experiment_id,
                        "Query_ID": j
                    })
            break

        # Write the .query and .evid files for this query 
        with open(
            f"{data_path}/{dataset}/{dataset}.query", mode='w'
        ) as f:
            f.write(q)
        with open(
            f"{data_path}/{dataset}/{dataset}.evid", mode='w'
        ) as f:
            f.write(e)

        # Get the probability of the evidence
        evid_info = e.split()
        index = 1
        evidence = Evidence()
        while index < len(evid_info):
            var_id = int(evid_info[index])
            val = int(evid_info[index + 1])
            evidence[spn.scope()[var_id]] = [val]
            index += 2
        p_evid = spn.log_value(evidence)

        # Run the queries through the MAP methods, store results
        if "WMB" in methods:
            try:
                print(f"Starting query {i} (WMB), remaining budget ~{remaining_dataset:.1f}s")
                start = time.perf_counter()
                q_var_indices = [int(var) for var in q.split()[1:]]
                q_vars = [spn.scope()[ind] for ind in q_var_indices]
                result = merlin(
                    evidence_file=f"{data_path}/{dataset}/{dataset}.evid",
                    query_file=f"{data_path}/{dataset}/{dataset}.query",
                    uai_file=f"{data_path}/{dataset}/{dataset}.uai",
                    algorithm='wmb',
                    ibound=10,
                    iterations=2,
                    query_vars=spn.scope(),
                    timeout=max(1, int(remaining_dataset)),
                )
                if result == 'timeout':
                    print('Timeout triggered (WMB), prob of 0 assigned')
                    wmb_prob = 0.0
                    wmb_est = evidence
                else:
                    wmb_est = result[3]
                    wmb_prob = spn.log_value(wmb_est) - p_evid
                    wmb_prob = np.exp(wmb_prob)
                wmb_time = time.perf_counter() - start
                print(f"Query runtime (WMB): {wmb_time}")
                results.append({
                    "Date": datetime_str,
                    "Dataset": dataset,
                    "Query": q_var_indices,
                    "Method": "WMB Elimination",
                    "MAP Estimate": "NA",
                    "MAP Probability": wmb_prob,
                    "Runtime": wmb_time,
                    "Query Proportion": q_percent,
                    "Evid Proportion": e_percent,
                    "Experiment ID": experiment_id,
                    "Query_ID": i
                })
                print(f"WMB:           {wmb_prob:.4g}")
                print()
            except Exception as error:
                print(f"WMB failed with error {error}")
                print(f"Error type: {type(error).__name__}")
                run_success = False
                break

        if "AAOBF" in methods:
            try:
                # Recompute remaining budget just before AAOBF
                elapsed_dataset = time.perf_counter() - dataset_start
                remaining_dataset = time_budget - elapsed_dataset
                if remaining_dataset <= 0:
                    # No time left for AAOBF on this and later queries
                    for j in range(i, len(queries) + 1):
                        q_line = queries[j - 1]
                        q_var_indices = [int(var) for var in q_line.split()[1:]]
                        results.append({
                            "Date": datetime_str,
                            "Dataset": dataset,
                            "Query": q_var_indices,
                            "Method": "AAOBF",
                            "MAP Estimate": "NA",
                            "MAP Probability": 0.0,
                            "Runtime": 0.0,
                            "Query Proportion": q_percent,
                            "Evid Proportion": e_percent,
                            "Experiment ID": experiment_id,
                            "Query_ID": j
                        })
                    break

                print(f"Starting query {i} (AAOBF), remaining budget ~{remaining_dataset:.1f}s")
                start = time.perf_counter()
                q_var_indices = [int(var) for var in q.split()[1:]]
                q_vars = [spn.scope()[ind] for ind in q_var_indices]
                result = merlin(
                    evidence_file=f"{data_path}/{dataset}/{dataset}.evid",
                    query_file=f"{data_path}/{dataset}/{dataset}.query",
                    uai_file=f"{data_path}/{dataset}/{dataset}.uai",
                    algorithm='any-aaobf',
                    ibound=10,
                    iterations=2,
                    query_vars=spn.scope(),
                    timeout=max(1, int(remaining_dataset)),
                )
                if result == 'timeout':
                    print('Timeout triggered (AAOBF), prob of 0 assigned')
                    aaobf_prob = 0.0
                else:
                    aaobf_prob = result - p_evid
                    aaobf_prob = np.exp(aaobf_prob)
                aaobf_time = time.perf_counter() - start
                print(f"Query runtime (AAOBF): {aaobf_time}")
                results.append({
                    "Date": datetime_str,
                    "Dataset": dataset,
                    "Query": q_var_indices,
                    "Method": "AAOBF",
                    "MAP Estimate": "NA",  # AAOBF doesn't output assignment
                    "MAP Probability": aaobf_prob,
                    "Runtime": aaobf_time,
                    "Query Proportion": q_percent,
                    "Evid Proportion": e_percent,
                    "Experiment ID": experiment_id,
                    "Query_ID": i
                })
                print(f"AAOBF:         {aaobf_prob:.4g}")
                print()
            except Exception as error:
                print(f"AAOBF failed with error {error}")
                print(f"Error type: {type(error).__name__}")
                run_success = False
                break

        # After finishing this query (for all methods), check if budget is exhausted
        elapsed_dataset = time.perf_counter() - dataset_start
        remaining_dataset = time_budget - elapsed_dataset
        if remaining_dataset <= 0:
            # Fill zeros for all *later* queries
            for j in range(i + 1, len(queries) + 1):
                q_line = queries[j - 1]
                q_var_indices = [int(var) for var in q_line.split()[1:]]
                if "WMB" in methods:
                    results.append({
                        "Date": datetime_str,
                        "Dataset": dataset,
                        "Query": q_var_indices,
                        "Method": "WMB Elimination",
                        "MAP Estimate": "NA",
                        "MAP Probability": 0.0,
                        "Runtime": 0.0,
                        "Query Proportion": q_percent,
                        "Evid Proportion": e_percent,
                        "Experiment ID": experiment_id,
                        "Query_ID": j
                    })
                if "AAOBF" in methods:
                    results.append({
                        "Date": datetime_str,
                        "Dataset": dataset,
                        "Query": q_var_indices,
                        "Method": "AAOBF",
                        "MAP Estimate": "NA",
                        "MAP Probability": 0.0,
                        "Runtime": 0.0,
                        "Query Proportion": q_percent,
                        "Evid Proportion": e_percent,
                        "Experiment ID": experiment_id,
                        "Query_ID": j
                    })
            break

    if run_success:
        results_dt = pd.DataFrame(results)
        if no_results_file is False:
            file_exists = os.path.isfile(results_filename)
            results_dt.to_csv(results_filename, mode='a', header=not file_exists, index=False)
        else:
            results_dt.to_csv(f"{data_path}/{dataset}/{dataset}_results.csv", index=False)