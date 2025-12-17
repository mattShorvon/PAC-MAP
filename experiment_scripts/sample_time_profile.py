from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample, sample_parallel
import numpy as np
from spn.structs import Variable
from spn.io.file import from_file
import time
from pathlib import Path
import os

# Load the data
dataset = 'mushrooms'
data_path = 'small_datasets'
spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
q_percent = 0.4
e_percent = 0.6
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

# Test the time taken for the sampling
start = time.time()
for evid in evidences:
    sample(spn, num_samples=1000, evidence=evid)
time_sequential = time.time() - start

# Get the number of cores, and set the number of jobs
num_cores = os.cpu_count()
num_jobs = max(1, num_cores - 1)
start = time.time()
for evid in evidences:  
    sample_parallel(spn, num_samples=1000, evidence=evid, n_jobs=num_jobs)
time_parallel = time.time() - start

print(f"Time taken by sequential sampling: {time_sequential}")
print(f"Time taken by parallel sampling with {num_jobs} cores: {time_parallel}")