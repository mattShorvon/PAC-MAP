import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from spn.actions.learn import Learn
from spn.learn import gens
import argparse

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

args = parser.parse_args()
use_all = args.all_datasets
data_path = args.data_path
if use_all:
    datasets = sorted(
        [folder.name for folder in os.scandir(data_path) if folder.is_dir()]
    )
else:
    datasets = args.datasets

print(f"Datasets: {datasets}")

for dataset in datasets:
    # Set up the SPN
    print(f"Running benchmark on dataset {dataset}")
    print("Learning SPN ...")
    kclusters = 3
    pval = 0.5
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