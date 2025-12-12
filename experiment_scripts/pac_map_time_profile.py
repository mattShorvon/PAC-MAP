from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample
from spn.actions.likelihood import ll_from_data
import numpy as np
from spn.structs import Variable
from spn.io.file import from_file
import time
from pathlib import Path

# For working out why pac-map is so slow on the cluster
# To run: kernprof -l -v experiment_scripts/pac_map_time_profile.py && rm pac_map_time_profile.py.lprof

@profile
def pac_map_timed(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
        err_tol: float = 0.05,
        fail_prob: float = 0.05,
        ) -> Tuple[Evidence, float]:
    
    candidate_list = []
    probs = []
    seen_hashes = set()
    m = 0
    M = float('inf')
    p_hat = float('-inf')
    p_tick = 0
    q_hat = None
    p_evid = spn.value(evidence)
    
    # Track timing
    time_sampling = 0
    time_filtering = 0
    time_likelihood = 0
    time_updating = 0
    
    def sample_evid_to_tuple(evid):
        return tuple(sorted((var.id, vals[0]) for var, vals in evid.items())) 

    iteration = 0
    while m < M:
        iteration += 1
        print(f"Iteration {iteration}, m={m}, M={M:.2f}, candidates={len(candidate_list)}")
        
        m += batch_size

        # Time sampling
        t0 = time.time()
        new_samples = sample(spn, num_samples=batch_size, evidence=evidence)
        time_sampling += time.time() - t0
        
        # Time filtering
        t0 = time.time()
        unseen_samples = []
        for sample_dict in new_samples:
            filtered_sample = Evidence({var: vals for var, vals in sample_dict.items() 
                                            if var not in marginalized})
            sample_hash = sample_evid_to_tuple(filtered_sample)
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unseen_samples.append(filtered_sample)
                candidate_list.append(filtered_sample)
        time_filtering += time.time() - t0
        print(f"  New samples: {len(new_samples)}, Unseen: {len(unseen_samples)}")
        
        # Time likelihood computation
        t0 = time.time()
        if unseen_samples:
            new_probs = np.exp(ll_from_data(spn, unseen_samples)) / p_evid
            probs.extend(new_probs)
        time_likelihood += time.time() - t0

        # Time updating
        t0 = time.time()
        if max(probs) > p_hat:
            p_hat = max(probs)
            q_hat_idx = np.argmax(probs)
            q_hat = candidate_list[q_hat_idx]
        
        p_tick = 1 - sum(probs)
        if p_tick <= p_hat/(1 - err_tol):
            M = 0
        else:
            M = np.log(fail_prob) / np.log((1 - p_hat)/(1 - err_tol))
        time_updating += time.time() - t0
        
        print(f"  p_hat={p_hat:.6f}, p_tick={p_tick:.6f}")
    
    print(f"\n=== Timing Summary ===")
    print(f"Sampling: {time_sampling:.2f}s")
    print(f"Filtering: {time_filtering:.2f}s")
    print(f"Likelihood: {time_likelihood:.2f}s")
    print(f"Updating: {time_updating:.2f}s")
    print(f"Total iterations: {iteration}")

    return [q_hat, p_hat]

if __name__ == "__main__":
    # In your benchmark.py, wrap the call:
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    # Load the data
    dataset = 'audiology'
    data_path = 'small_datasets'
    spn = from_file(Path(f"{data_path}/{dataset}/{dataset}.spn"))
    q_percent = 0.1
    e_percent = 0.9
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


    # Your pac_map call here
    results = []
    for q, e in zip(queries, evidences):
        m = [var for var in spn.scope() if var not in q and var not in e]
        pac_map_est, pac_map_prob = pac_map_timed(
                spn, e, m, batch_size=100, err_tol=0.01, fail_prob=0.01
            )
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20) 