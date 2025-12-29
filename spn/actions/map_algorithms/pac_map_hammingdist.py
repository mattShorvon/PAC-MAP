from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample_multiproc
from spn.actions.likelihood import likelihood_multiproc
import numpy as np
from spn.structs import Variable
from itertools import combinations
from itertools import product
import copy
from pathlib import Path
from spn.actions.condition import condition_spn
from spn.io.file import to_file
import time
import os

def sample_evid_to_tuple(evid):
    """Convert sample in Evidence() dictionary format to hashable tuple 
    (assumes 1 val per variable)."""
    return tuple(sorted((var.id, vals[0]) for var, vals in evid.items())) 

def bits_to_evid(bitstring, variables):
    neighbour = Evidence()
    for var, val in zip(variables, bitstring):
        neighbour[var] = [val]
    return neighbour


def explore(spn_path, batch_size, evidence, m, marginalized, n_jobs):
    # Draw new samples from P(Q | E)
    new_samples = sample_multiproc(spn_path, 
                                   num_samples=batch_size,
                                   evidence=evidence,
                                   marginalized=marginalized,
                                   n_jobs=n_jobs)
    
    # Update number of samples taken so far
    m += batch_size
    return new_samples, m

def exploit(q_hat, evidence, h_radius=1):
    """
    Generate all neighbours within Hamming distance h_radius from q_hat.
    """
    
    # Filter out evidence variables
    q_hat_filtered = Evidence(
        {var: vals for var, vals in q_hat.items() if var not in evidence}
    )
    
    neighbours = []
    variables = list(q_hat_filtered.keys())
    
    # For each subset of variables to flip (size = h_radius)
    for vars_to_flip in combinations(variables, h_radius):
        # Build list of possible alternative values for each variable to flip
        value_options = []
        for var in vars_to_flip:
            current_val = q_hat_filtered[var][0]
            # All values except the current one
            alternative_vals = [v for v in range(var.n_categories) if v != current_val]
            value_options.append(alternative_vals)
        
        # Generate all combinations of alternative values
        for value_combination in product(*value_options):
            # Create a new neighbour with these flipped values
            neighbour = copy.deepcopy(Evidence(q_hat_filtered))
            for var, new_val in zip(vars_to_flip, value_combination):
                neighbour[var] = [new_val]
            
            # Add the evidence vars back into the neighbour and add it to list
            neighbour.merge(evidence)
            neighbours.append(neighbour)
    
    return neighbours

def exploit_bitwise(q_hat, evidence, h_radius=1):
    """
    Generate all neighbours within Hamming distance h_radius from q_hat.
    Uses faster bitwise XOR operations but only works for binary data
    """
    # Filter out evidence variables
    q_hat_filtered = Evidence(
        {var: vals for var, vals in q_hat.items() if var not in evidence}
    )
    
    # Sort variables by ID for consistent ordering
    variables = sorted(q_hat_filtered.keys(), key=lambda v: v.id)
    n_vars = len(variables)
    
    # Convert to bitstring
    bitstring = 0
    for i, var in enumerate(variables):
        if q_hat_filtered[var][0] == 1:
            bitstring |= (1 << i)
    
    neighbours = []
    
    if h_radius == 1:
        # Fast path: flip each bit once
        for i in range(n_vars):
            neighbour_bits = bitstring ^ (1 << i)
            
            # Convert back to Evidence
            neighbour = Evidence()
            for j, var in enumerate(variables):
                bit_value = (neighbour_bits >> j) & 1
                neighbour[var] = [bit_value]
            
            neighbours.append(neighbor)
    else:
        # General case for h_radius > 1
        from itertools import combinations
        for positions in combinations(range(n_vars), h_radius):
            flip_mask = sum(1 << pos for pos in positions)
            neighbor_bits = bitstring ^ flip_mask
            
            neighbor = Evidence()
            for j, var in enumerate(variables):
                bit_value = (neighbor_bits >> j) & 1
                neighbor[var] = [bit_value]
            
            neighbours.append(neighbor)
    
    return neighbours

def pac_map_hamming(
        spn: SPN, 
        spn_path: Path,
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 100,
        h_radius: int = 1,
        err_tol: float = 0.05,
        fail_prob: float = 0.05,
        sample_cap: int = 50000, 
        n_jobs: int = -1
        ) -> Tuple[Evidence, float]:
    
        # Condition SPN on evidence once at the start
    if evidence and len(evidence) > 0:
        # Condition spn
        conditioned_spn = condition_spn(spn, evidence, marginalized)
        
        # Save to temporary file for multiprocessing
        # with tempfile.NamedTemporaryFile(mode='wb', suffix='.spn', delete=False) as f:
        #     conditioned_spn_path = Path(f.name)
        timestamp = int(time.time())
        conditioned_spn_path = spn_path.parent / f"{spn_path.stem}_conditioned_{timestamp}_{os.getpid()}.spn"
        to_file(conditioned_spn, conditioned_spn_path)
        
        working_path = conditioned_spn_path
        sampling_evidence = None
    else:
        # No evidence, use original SPN
        conditioned_spn_path = None
        working_path = spn_path
        sampling_evidence = None

    candidate_list = []
    probs = []
    seen_hashes = set()
    m = 0
    M = float('inf')
    p_hat = float('-inf')
    p_tick = 0
    q_hat = None

    try:
        while m < M:
            # Draw new samples
            new_samples, m = explore(working_path, batch_size, sampling_evidence, m, marginalized, n_jobs)

            # Add samples that haven't been seen before to candidate_list 
            # (uses hashset for O(1) membership check)
            unseen_samples = []
            for sample_dict in new_samples:
                filtered_sample = Evidence({var: vals for var, vals in sample_dict.items() 
                                                if var not in marginalized})
                sample_hash = sample_evid_to_tuple(filtered_sample)
                if sample_hash not in seen_hashes:
                    seen_hashes.add(sample_hash)
                    unseen_samples.append(filtered_sample)
                    candidate_list.append(filtered_sample)
            
            # Compute likelihoods for new, unseen samples
            new_probs = np.exp(
                likelihood_multiproc(working_path, unseen_samples, n_jobs=n_jobs)
            )
            probs.extend(new_probs)

            # Check if you need to update the best candidate
            if max(probs) > p_hat:
                p_hat = max(probs)
                q_hat_idx = np.argmax(probs)
                q_hat = candidate_list[q_hat_idx]
            
            # Search in a hamming ball around the top sample
            new_samples = exploit(q_hat, evidence, h_radius)

            # Add samples that haven't been seen before to candidate_list 
            # (uses hashset for O(1) membership check)
            unseen_samples = []
            for sample_dict in new_samples:
                filtered_sample = Evidence({var: vals for var, vals in sample_dict.items() 
                                                if var not in marginalized})
                sample_hash = sample_evid_to_tuple(filtered_sample)
                if sample_hash not in seen_hashes:
                    seen_hashes.add(sample_hash)
                    unseen_samples.append(filtered_sample)
                    candidate_list.append(filtered_sample)
            
            # Compute likelihoods for new, unseen samples
            if unseen_samples:
                new_probs = np.exp(
                    likelihood_multiproc(working_path, unseen_samples, n_jobs=n_jobs)
                )
                probs.extend(new_probs)

                # Check if you need to update the best candidate
                if max(probs) > p_hat:
                    p_hat = max(probs)
                    q_hat_idx = np.argmax(probs)
                    q_hat = candidate_list[q_hat_idx]
            
            # Check if you can issue the PAC certificate, currently doing this in 
            # prob space rather than log lik space
            p_tick = 1 - sum(probs)
            if p_tick <= p_hat/(1 - err_tol):
                M = 0
            else:
                # M = np.log(fail_prob) / np.log((1 - p_hat)/(1 - err_tol)) wrong version
                # M = np.log(fail_prob) / np.log(1 - (p_hat/(1 - err_tol))) correct exact version
                M = np.log(1/fail_prob)/(p_hat/(1 - err_tol)) # correct approximate version
            
            # Check if the sample_cap has been hit, stop early and return current
            # delta and epsilon pareto front if so (not sure how to return them yet)
            if m >= sample_cap:
                print("Sample cap reached!")
                epsilon = np.linspace(0, 1 - p_hat - 1e-6, 200)
                delta = (1 - p_hat / (1 - epsilon)) ** M
                return [q_hat, p_hat]
        
    finally:
        # Cleanup: delete temporary conditioned SPN file
        if conditioned_spn_path is not None and conditioned_spn_path.exists():
            conditioned_spn_path.unlink()

    return [q_hat, p_hat]

def pac_map_hamming_eta(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
        h_radius: int = 1,
        eta: float = 0.1,
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

    while m < M:
        actions = ['explore', 'exploit']
        action = np.random.choice(actions, size=1, p=[1 - eta, eta]).tolist()[0]

        if action == 'explore':
            new_samples, m = explore(spn, batch_size, evidence, m)
        
        if action == 'exploit':
            if m == 0:
                # Can't exploit if no samples have been drawn yet
                new_samples, m = explore(spn, batch_size, evidence, m)
            else:
                new_samples = exploit(q_hat, evidence, h_radius)

        
        # Add samples that haven't been seen before to candidate_list 
        # (uses hashset for O(1) membership check)
        unseen_samples = []
        for sample_dict in new_samples:
            filtered_sample = Evidence({var: vals for var, vals in sample_dict.items() 
                                            if var not in marginalized})
            sample_hash = sample_evid_to_tuple(filtered_sample)
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unseen_samples.append(filtered_sample)
                candidate_list.append(filtered_sample)
        
        # Compute likelihoods for new, unseen samples
        new_probs = np.exp(ll_from_data(spn, unseen_samples)) / p_evid
        probs.extend(new_probs)

        # Check if you need to update the best candidate
        if max(probs) > p_hat:
            p_hat = max(probs)
            q_hat_idx = np.argmax(probs)
            q_hat = candidate_list[q_hat_idx]
        
        # Check if you can issue the PAC certificate, currently doing this in 
        # prob space rather than log lik space
        p_tick = 1 - sum(probs)
        if p_tick <= p_hat/(1 - err_tol):
            M = 0
        else:
            M = np.log(fail_prob) / np.log((1 - p_hat)/(1 - err_tol))

    return [q_hat, p_hat]