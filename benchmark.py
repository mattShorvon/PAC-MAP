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
from lbp import lbp
from spn.utils.graph import full_binarization
from spn.utils.evidence import Evidence
import argparse
import pandas as pd
from datetime import datetime

# Check command line arguments and initialise variables
parser = argparse.ArgumentParser(description='Benchmark SPN MAP algorithms')
parser.add_argument('-d', '--datasets', nargs='+', required=True,
                    help='Dataset names separated by space (e.g., iris nltcs)')
parser.add_argument('-m', '--methods', nargs='+', required=True,
                    help='MAP algs to run, separated by space (e.g. MP AMP)')
parser.add_argument('--learn', action='store_true',
                    help='Learn SPN from scratch')
parser.add_argument('--no-learn', dest='learn', action='store_false',
                    help='Use existing SPN')
parser.add_argument('--file-mode', choices=['.map','.query/.evid'], 
                    default='.map',
                    help='Format of query/evidence files')
parser.add_argument('--data-path', default='test_inputs', 
                    help='Path to folder with all subfolders of spns, ' \
                    'queries and evidence in them')
parser.add_argument('--no-res-file', action="store_true",
                    help="No single file to write the results to, will write" \
                    "to many individual files in each dataset's subfolder instead")
parser.add_argument('--results-file', default='benchmark_results.csv',
                    help='Path to file to store results in, if storing in ' \
                    'a single results file')
parser.set_defaults(learn=False)

args = parser.parse_args()

datasets = args.datasets
methods = args.methods
learn_spn = args.learn
query_evid_filemode = args.file_mode
data_path = args.data_path
no_results_file = args.no_res_file
results_filename = args.results_file

print(f"Datasets: {datasets}")
print(f"MAP methods being run: {methods}")
print(f"Learning SPN from scatch: {learn_spn}")

datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

# Run the experiment
for dataset in datasets:
    # Set up the SPN
    if learn_spn:
        print("Learning SPN ...")
        kclusters = 10
        pval = 0.1
        data = PartitionedData(
            Path(f"{data_path}/{dataset}/{dataset}-train.data", 1.0)
        )
        spn = gens(data.scope, data.training_data, kclusters, pval)
        spn.root = True
        print("SPN learned:", spn.vars(), "vars and", spn.arcs(), "arcs")
    else:
        spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
        print(f"SPN loaded: {spn.vars()} vars and {spn.arcs()} arcs")

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
        if "MP" in methods:
            mp_est, _ = max_product_with_evidence_and_marginals(
                spn, e, m
            )
            mp_prob = spn.value(mp_est)
            
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "Max Product",
                "MAP Estimate": str({var.id: mp_est[var] for var in q}),
                "MAP Probability": mp_prob,
            })
        if "AMP" in methods:
            amp_est, _ = argmax_product_with_evidence_and_marginalized(
                spn, e, m
            )
            amp_prob = spn.value(amp_est)
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "ArgMax Product",
                "MAP Estimate": str({var.id: amp_est[var] for var in q}),
                "MAP Probability": amp_prob,
            })
        if "MS" in methods:
            ms_est, _ = max_search(
                spn,
                forward_checking,
                time_limit=600,
                marginalized_variables=m,
                evidence=e,
            )
            ms_prob = spn.value(ms_est)
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "Max Search",
                "MAP Estimate": str({var.id: ms_est[var] for var in q}),
                "MAP Probability": ms_prob,
            })
        if "HBP" in methods:
            spn_bin = full_binarization(spn)
            spn.fix_scope()
            spn.fix_topological_order()
            hbp_est = lbp(spn_bin, e, m, num_iterations=5)
            hbp_prob = spn_bin.value(hbp_est)
            results.append({
                "Date": datetime_str,
                "Dataset": dataset,
                "Query": str([query.id for query in q]),
                "Method": "Hybrid Belief-Propagation",
                "MAP Estimate": str({var.id: hbp_est[var] for var in q}),
                "MAP Probability": hbp_prob,
            })
    results_dt = pd.DataFrame(results)
    if no_results_file is False:
        results_dt.to_csv(results_filename, index=False)
    else:
        results_dt.to_csv(f"{data_path}/{dataset}/{dataset}_results.csv", index=False)
