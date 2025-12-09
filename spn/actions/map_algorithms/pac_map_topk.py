from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample
from spn.actions.likelihood import ll_from_data
import numpy as np
from spn.structs import Variable
import copy
import pandas as pd
from collections import defaultdict



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

def exploit(candidates, evidence, n_batch=10):
    """
    Take the existing candidates, sort them by probability and take the top k. 
    Then for each feature, use the k candidates to find the empirical prob that
    the feature has each value, then draw from a categorical distribution with
    probs given by these empirical probs.
    """
    # Filter out evidence vars
    filtered_candidates = [
        Evidence({var: vals for var, vals in cand.items() if var not in evidence})
        for cand in candidates
    ]

    # Create a hash map of the number of times each value of each var appears
    vars = filtered_candidates[0].variables
    value_counts = defaultdict(lambda: defaultdict(int))
    for cand in filtered_candidates:
        for var in vars:
            val = cand[var][0]
            value_counts[var][val] += 1
    
    # Convert the counts to probabilities
    for var in vars:
        for val in value_counts[var]:
            value_counts[var][val] = value_counts[var][val]/len(filtered_candidates)
    
    # Use these probabilities to sample n_batch new samples. 
    new_samples = []
    for _ in range(n_batch):
        sample = Evidence()
        for var in vars:
            vals = list(value_counts[var].keys())
            probs = list(value_counts[var].values())
            sample_val = np.random.choice(a=vals, size=1, p=probs)[0]
            sample[var] = [sample_val]
        sample.merge(evidence)
        new_samples.append(sample)
    
    return new_samples

def exploit_binary(candidates):
    """
    Take the existing candidates, sort them by probability and take the top k. 
    Then for each feature, use the k candidates to find the empirical prob that
    the feature has value 1, then draw from a bernoulli distribution with
    probs given by these empirical probs.
    """
    # Filter out evidence variables
    q_hat_filtered = Evidence(
        {var: vals for var, vals in q_hat.items() if var not in evidence}
    )
    
    neighbours = []
    return neighbours

def pac_map_topk(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 100,
        k: int = 10,
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

        # Run the TopK routine to exploit information from the current candidates
        sorted_cand_ids = np.argsort(probs)[::-1][:k]
        topk_cands = [candidate_list[i] for i in sorted_cand_ids]
        new_samples = exploit(topk_cands, evidence, batch_size)

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
        
        if unseen_samples:
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

def pac_map_topk_eta(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
        k: int = 10,
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
                candidate_ids = [i for i, _ in enumerate(candidate_list)]
                probs_df = pd.DataFrame()
                sorted_cand_ids = np.argsort(probs)[::-1][:k]
                topk_cands = [candidate_list[i] for i in sorted_cand_ids]
                new_samples = exploit(topk_cands, evidence, batch_size)

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