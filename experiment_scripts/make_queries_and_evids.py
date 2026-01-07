import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import random
from spn.io.file import from_file
from spn.utils.evidence import Evidence
from spn.actions.sample import sample

# EXAMPLE: python make_queries_and_evids.py -dp benchmark_datasets -all -q 0.4 -e 0.4
# Check command line args and initialise variables
parser = argparse.ArgumentParser(description='Make .map files of queries and ' \
'evidences')
parser.add_argument('-all','--all-datasets', action='store_true',
                    help="Iterate through all dataset subfolders" \
                    "(containing queries, evidences and spns) in the specified"\
                    "data path, or just use specific ones?")
parser.add_argument('-dp', '--data-path', default='benchmark_datasets', 
                    help='Path to folder with all subfolders of spns, ' \
                    'queries and evidence in them')
parser.add_argument('-d', '--datasets', nargs='+',
                    help='Dataset names separated by space (e.g., iris nltcs)')
parser.add_argument('-n', '--num-queries', default=10,
                    help="Number of query and evidence pairs to create")
parser.add_argument('-q', '--q-percent', type=float, default=0.3, 
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float, default=0.3,
                    help="Proportion of evidence variables")

args = parser.parse_args()
use_all = args.all_datasets
data_path = args.data_path
if use_all:
    datasets = sorted(
        [folder.name for folder in os.scandir(data_path) if folder.is_dir()]
    )
else:
    datasets = args.datasets
num_queries = args.num_queries
q_percent = args.q_percent
e_percent = args.e_percent

print(f"Datasets: {datasets}")

for dataset in datasets:
    # Training data files come in two different naming formats, so we have to
    # try both
    try:
        with open(f"{data_path}/{dataset}/{dataset}-train.data") as f:
            for line in f:
                line = line.split()
                if line[0] == 'var':
                    continue
                else:
                    line = line[0].split(',')
                    num_features = len(line)
                    break
    except:
        try:
            with open(f"{data_path}/{dataset}/{dataset}.train.data") as f:
                for line in f:
                    line = line.split()
                    if line[0] == 'var':
                        continue
                    else:
                        num_features = len(line)
                        break
        except Exception as error:
            print(f"Failed to load training data: {error}")
            print(f"Error type: {type(error).__name__}")
            continue
    print(f"Number of features: {num_features}")

    # Load the spn, to be used for making sure the random evidence is not 0 prob
    try:
        spn = from_file(Path(f"{data_path}/{dataset}/{dataset}_0.1q_0.9e.spn"))
        print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")
    except FileNotFoundError as error:
        print(".spn file not found, check the file path or that the spn exists")
        print(error)
        print("moving on to the next dataset")

    # Write the query and evidence lines
    lines = []
    for i in range(num_queries):
        num_q_vars = np.floor(num_features * q_percent)
        num_e_vars = np.floor(num_features * e_percent)
        if num_q_vars == 0.0:
            num_q_vars = 1
        if num_e_vars == 0.0:
            num_e_vars = 1
        q_var_ids = random.sample(range(num_features), int(num_q_vars))
        query_vars = [var for var in spn.scope() if var.id in q_var_ids]
        remaining = [i for i in range(num_features) if i not in q_var_ids]
        test_evid_prob = float('-inf')
        evidences_tried = 0
        while test_evid_prob == float('-inf'):
            e_var_ids = random.sample(remaining, min(int(num_e_vars), len(remaining)))
            e_values = random.choices([0, 1], k=len(e_var_ids))
            test_evid = Evidence()
            for i, var in enumerate(e_var_ids):
                test_evid[spn.scope()[var]] = [e_values[i]]
            test_evid_prob = spn.log_value(test_evid)
            evidences_tried += 1
            if evidences_tried > 10:
                test_evid = sample(spn, 1, None, marginalized=query_vars)[0]
                print("Couldn't find non-zero prob evidence randomly, "
                      "resorted to sampling")
                test_evid_prob = spn.log_value(test_evid)
                e_var_ids = [var.id for var in test_evid.variables]
                e_values = [test_evid[var][0] for var in test_evid.variables]

        print(f"Found non-zero prob evidence after {evidences_tried} attempts")
        e_line = ' '.join(
            [f"{var_id} {val}" for var_id, val in zip(e_var_ids, e_values)]
        )
        q_line = ' '.join(map(str, q_var_ids))
        lines.append(q_line)
        lines.append(e_line)

    # Write to .map file
    with open(
        f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e.map", 'w'
        ) as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Created {dataset}.map with {num_queries} query/evidence pairs") 

