import os
import pandas as pd
import numpy as np
import argparse
from spn.io.file import from_file
from spn.actions.map_algorithms.pac_map import pac_map
from datetime import datetime
from pathlib import Path
import time
from spn.utils.evidence import Evidence

parser = argparse.ArgumentParser(
    description="PAC-MAP batch-size vs runtime experiment parameters"
)
parser.add_argument('-dp', '--data-path', default='test_inputs', 
                    help='Path to folder with all subfolders of spns, ' \
                    'queries and evidence in them')
parser.add_argument('-d', '--datasets', nargs='+',
                    help='Dataset names separated by space (e.g., iris nltcs)')
parser.add_argument('-bs', '--batch-sizes', nargs='+',
                    help='Batch sizes to test, separated by a space')
parser.add_argument('-q', '--q-percent', type=float,
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float,
                    help="Proportion of evidence variables")
parser.add_argument('--results-file', default='batchsz_vs_runtime_results.csv',
                    help='Path to file to store results in, if storing in ' \
                    'a single results file')
parser.add_argument('-dt', '--date',
                    default=datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                    help='Date and time of experiment')

# Parse the input arguments
args = parser.parse_args()
data_path = args.data_path
datasets = args.datasets
batch_sizes = args.batch_sizes
if args.q_percent and args.e_percent:
    q_percent = args.q_percent
    e_percent = args.e_percent
else: 
    q_percent, e_percent = None, None
results_filename = args.results_file
datetime_str = args.date

for dataset in datasets:
    # Load the spn
    try:
        spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
        print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")
    except FileNotFoundError as error:
        print(".spn file doesn't exist in this subfolder")
        print(error)
        continue

    # Load the queries and evidences
    queries, evidences = [], []
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

    # Run the experiment 
    results = []
    for batch_sz in batch_sizes:
        for q, e in zip(queries, evidences):
            m = [var for var in spn.scope() if var not in q and var not in e]
            start = time.perf_counter()
            pac_map_est, pac_map_prob = pac_map(
                spn, e, m
            )
            # pac_map_prob = spn.value(pac_map_est)
            pac_map_time = time.perf_counter() - start
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Batch Size": batch_sz,
                "MAP Probability": pac_map_prob,
                "Runtime": pac_map_time,
                "Query Proportion": q_percent,
                "Evid Proportion": e_percent
            })
            print(f"Batch Size:        {batch_sz}")
            print(f"Runtime:           {pac_map_time:.4g}")
            print("MAP Est:", ' '.join([str(pac_map_est[v]) for v in q]))
            print()
    results_dt = pd.DataFrame(results)
    file_exists = os.path.isfile(results_filename)
    results_dt.to_csv(results_filename, mode='a', header=not file_exists, index=False)