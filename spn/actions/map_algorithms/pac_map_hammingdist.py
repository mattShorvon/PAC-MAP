from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample
from spn.actions.likelihood import ll_from_data
import numpy as np
from spn.structs import Variable
from itertools import combinations
from itertools import product
import copy


def sample_evid_to_tuple(evid):
    """Convert sample in Evidence() dictionary format to hashable tuple 
    (assumes 1 val per variable)."""
    return tuple(sorted((var.id, vals[0]) for var, vals in evid.items())) 

def bits_to_evid(bitstring, variables):
    neighbour = Evidence()
    for var, val in zip(variables, bitstring):
        neighbour[var] = [val]
    return neighbour


def explore(spn, batch_size, evidence, m):
    # Draw new samples from P(Q | E)
    new_samples = sample(spn, 
                        num_samples=batch_size,
                        evidence=evidence)
    
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

def pac_map_hamming_no_eta(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
        h_radius: int = 1,
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
        # Draw new samples
        new_samples, m = explore(spn, batch_size, evidence, m)

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