import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from spn.io.file import from_file
from spn.actions.map_algorithms.argmax_product import (
    argmax_product_with_evidence_and_marginalized,
)
from spn.actions.map_algorithms.pac_map_budget import pac_map_budget
from spn.actions.map_algorithms.pac_map import pac_map
from spn.actions.map_algorithms.pac_map_hammingdist import pac_map_hamming
from spn.utils.evidence import Evidence
import argparse
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Parse input args and initialise variables
parser = argparse.ArgumentParser(description="Amp+pac_map experiments params")
parser.add_argument('-dp', '--data-path', default='test_inputs', 
                    help='Path to folder with all subfolders of spns, ' \
                    'queries and evidence in them')
parser.add_argument('-d', '--datasets', nargs='+',
                    help='Dataset names separated by space (e.g., iris nltcs)')
parser.add_argument('-m', '--budgets', nargs="+", 
                    help='List of sample budgets to give to pac_map_budget')
parser.add_argument('-q', '--q-percent', type=float, required=True,
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float, required=True,
                    help="Proportion of evidence variables")
parser.add_argument('-id', '--experiment-id', default=1,
                    help='If running several experiments that you want to be ' \
                    'paired together, assign them the same id')
parser.add_argument('-dt', '--date',
                    default=datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                    help='Date and time of experiment')
parser.add_argument('--no-res-file', action="store_true",
                    help="No single file to write the results to, will write" \
                    "to many individual files in each dataset's subfolder instead")
parser.add_argument('--results-file', default='amp+pacmap_budget_results.csv',
                    help='Path to file to store results in, if storing in ' \
                    'a single results file')

args = parser.parse_args()
data_path = args.data_path
datasets = args.datasets
budgets = [int(budget) for budget in args.budgets]
q_percent = args.q_percent
e_percent = args.e_percent
experiment_id = args.experiment_id
datetime_str = args.date
no_results_file = args.no_res_file
results_filename = args.results_file
try:
    n_jobs = int(os.environ.get('SLURM_NTASKS'))
    print(f"On cluster, n_jobs set to {n_jobs}")
except TypeError:
    print("Not on cluster, n_jobs set to -2")
    n_jobs = -2 

# Run the experiment
for dataset in datasets:
    print(f"Running Experiment on dataset {dataset}")

    # Load the spn
    spn_path = Path(
        f"{data_path}/{dataset}/{dataset}_{q_percent}q_{e_percent}e.spn"
    )
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
    
    # Loop throught the evidence and query pairs
    results = []
    for q, e in zip(queries, evidences):
        for budget in budgets:
            print("Query       :", ' '.join([f"{v.id}" for v in q ]))
            print("Evidence    :", e)
            m = [var for var in spn.scope() if var not in q and var not in e]
            p_evid = spn.log_value(e)
            amp_est, _ = argmax_product_with_evidence_and_marginalized(
                spn, e, m
            )
            amp_est = Evidence({var: vals for var, vals in amp_est.items() if var not in m})
            amp_prob = spn.log_value(amp_est) - p_evid
            amp_prob = np.exp(amp_prob)
            pac_map_est, pac_map_prob, epsilon, delta = pac_map_budget(
                spn, spn_path, e, m, m=budget, n_jobs=n_jobs, 
                warm_start_cands=[amp_est], warm_start_probs=[amp_prob]
            )

            # Plot epsilon vs delta
            plt.figure(figsize=(10, 6))
            plt.plot(epsilon, delta, linewidth=2)
            plt.xlabel('Epsilon (ε)', fontsize=12)
            plt.ylabel('Delta (δ)', fontsize=12)
            plt.title(f'PAC Certificate: {dataset}, Budget={budget}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_path = Path(data_path) / dataset / f"pac_cert_budget{budget}_{time.time()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {plot_path}")


            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": ' '.join([f"{v.id}" for v in q ]),
                "Budget": budget,
                "AMP Prob": amp_prob,
                "PAC-MAP Prob": pac_map_prob
            })
            print(f"AMP prob: {amp_prob}")
            print(f"PAC-MAP prob: {pac_map_prob}")

    results_dt = pd.DataFrame(results)
    if no_results_file is False:
        file_exists = os.path.isfile(results_filename)
        results_dt.to_csv(results_filename, mode='a', header=not file_exists, index=False)
    else:
        results_dt.to_csv(f"{data_path}/{dataset}/{dataset}_results.csv", index=False)
