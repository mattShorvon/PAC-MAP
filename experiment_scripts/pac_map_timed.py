from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
import numpy as np
from spn.io.file import from_file
from pathlib import Path
import time
from spn.actions.map_algorithms.pac_map import pac_map
import os

# Load the data and initialise experiment vars
dataset = 'cwebkb'
data_path = '20-datasets'
spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
spn_path = Path(f"{data_path}/{dataset}/{dataset}.spn")
q_percent = 0.4
e_percent = 0.6
queries, evidences = [], []
print(f"Dataset: {dataset}")
print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS')}")
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

n_batch = 1000
n_cap = 50000
print(f"Running pac_map with a batch size of {n_batch} and sample cap of {n_cap}")
q = queries[0]
e = evidences[0]
m = [var for var in spn.scope() if var not in q and var not in e]
start = time.perf_counter()
pac_map_est, pac_map_prob = pac_map(
    spn, spn_path, e, m, batch_size=n_batch, err_tol=0.01, fail_prob=0.01,
    sample_cap=n_cap
)
pac_map_time = time.perf_counter() - start
print(f"Time taken by pac_map with batch size {n_batch} and cap {n_cap}: {pac_map_time}")

n_batch = 500
n_cap = 50000
start = time.perf_counter()
pac_map_est, pac_map_prob = pac_map(
    spn, spn_path, e, m, batch_size=n_batch, err_tol=0.01, fail_prob=0.01,
    sample_cap=n_cap
)
pac_map_time = time.perf_counter() - start
print(f"Time taken by pac_map with batch size {n_batch} and cap {n_cap}: {pac_map_time}")

