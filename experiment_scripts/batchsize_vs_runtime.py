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
import matplotlib.pyplot as plt

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
parser.add_argument('-dt', '--date',
                    default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    help='Date and time of experiment')

# Parse the input arguments
args = parser.parse_args()
data_path = args.data_path
datasets = args.datasets
batch_sizes = sorted([int(bs) for bs in args.batch_sizes])
if args.q_percent and args.e_percent:
    q_percent = args.q_percent
    e_percent = args.e_percent
else: 
    q_percent, e_percent = None, None
datetime_str = args.date
results_dir = "batchsz_vs_runtime_results"
datetime_folder = Path(results_dir) / datetime_str
datetime_folder.mkdir(parents=True, exist_ok=True)
results_filename = datetime_folder / f'results_{q_percent}q_{e_percent}e.csv'

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
                spn, e, m, batch_size=batch_sz
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

    # Plot the results
    runtime_results = results_dt.groupby(['Batch Size'])['Runtime'].agg(
        Mean_Runtime="mean",
        Std_Runtime="std"
    ).reset_index()
    prob_results = results_dt.groupby(['Batch Size'])['MAP Probability'].agg(
        Mean_Prob="mean",
        Std_Prob="std"
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.errorbar(
        runtime_results['Batch Size'],
        runtime_results['Mean_Runtime'],
        yerr=runtime_results['Std_Runtime'],
        marker='o',
        linestyle='-',
        capsize=5,
        capthick=2,
        label = 'Mean Runtime'
    )
    ax1.set_xscale('log')
    ax1.set_ylabel('Mean Runtime (seconds)')
    ax1.set_xlabel('Batch Size')
    ax1.set_title(f'Average Runtime vs Batch Size\n({q_percent}q {e_percent}e) on {dataset}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(
        prob_results['Batch Size'],
        prob_results['Mean_Prob'],
        yerr = prob_results['Std_Prob'],
        marker='o',
        linestyle='-',
        capsize=5,
        capthick=2,
        label='Mean MAP Probability'
    )
    ax2.set_xscale('log')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('MAP Probability', fontsize=12)
    ax2.set_title(f'Average MAP Probability vs Batch Size\n({q_percent}q {e_percent}e) on {dataset}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plot_filename = datetime_folder / f'batch_size_{dataset}_{q_percent}q_{e_percent}e.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()