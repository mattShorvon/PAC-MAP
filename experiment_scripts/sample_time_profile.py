from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample, sample_multithread, sample_multiproc
import numpy as np
from spn.structs import Variable
from spn.io.file import from_file
import time
from pathlib import Path
import os

# Load the data and initialise experiment vars
dataset = 'cwebkb'
data_path = '20-datasets'
spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
spn_path = Path(f"{data_path}/{dataset}/{dataset}.spn")
q_percent = 0.4
e_percent = 0.6
queries, evidences = [], []
run_sequential = False
run_multiproc = True
run_multithread = False
time_sequential, time_multithread, time_multiproc = None, None, None
print(f"Dataset: {dataset}")
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

# Test the time taken for sequential sampling
n = 5000
print(f"Starting test with {n} samples and {len(evidences)} evidences")
if run_sequential:
    start = time.time()
    for i, evid in enumerate(evidences):
        sample(spn, num_samples=n, evidence=evid)
        print(f"Completed iteration {i}")
    time_sequential = time.time() - start

# Get the number of cores, and set the number of jobs
# num_cores = os.cpu_count()
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS')}")
print(f"SLURM_CPUS_ON_NODE: {os.environ.get('SLURM_CPUS_ON_NODE')}")
num_cores = int(os.environ.get('SLURM_NTASKS', os.cpu_count()))
print(f"Num cores: {num_cores}")
num_jobs = max(1, num_cores) # don't set to num_cores -1 on the cluster

# Test how long it takes with multithreading
if run_multithread:
    start = time.time()
    for i,evid in enumerate(evidences):  
        sample_multithread(spn, num_samples=n, evidence=evid, n_jobs=num_jobs)
        print(f"Completed iteration {i}")
    time_multithread = time.time() - start

# Test how long it takes with multiprocessing
if run_multiproc:
    start = time.time()
    for i,evid in enumerate(evidences):  
        sample_multiproc(spn_path, num_samples=n, evidence=evid, n_jobs=num_jobs)
        print(f"Completed iteration {i}")
    time_multiproc = time.time() - start

print(f"Time taken by sequential sampling: {time_sequential}")
print(f"Time taken by multithread sampling with {num_jobs} cores: {time_multithread}")
print(f"Time taken by multiproc sampling with {num_jobs} cores: {time_multiproc}")

# Test how long it takes with half the cores
# num_cores_reduced = int(num_cores / 2)
# num_jobs = max(1, num_cores_reduced) # don't set to num_cores -1 on the cluster
# start = time.time()
# for evid in evidences:  
#     sample_multiproc(spn_path, num_samples=n, evidence=evid, n_jobs=num_jobs)
# time_parallel = time.time() - start

# print(f"Time taken by parallel sampling with {num_jobs} cores: {time_parallel}")