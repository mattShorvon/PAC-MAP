from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample
from spn.actions.likelihood import ll_from_data
import numpy as np
from spn.structs import Variable

def pac_map(
        spn: SPN, 
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
        err_tol: float = 0.05,
        fail_prob: float = 0.05
        ) -> Tuple[Evidence, float]:
    candidate_list = []
    probs = []
    seen_hashes = set()
    m = 0
    M = float('inf')
    p_hat = float('-inf')
    p_tick = 0
    q_hat = None
    p_evid = spn.log_value(evidence)
    
    def sample_evid_to_tuple(evid):
        """Convert sample in Evidence() dictionary format to hashable tuple 
        (assumes 1 val per variable)."""
        return tuple(sorted((var.id, vals[0]) for var, vals in evid.items())) 

    while m < M:
        # Update number of samples taken so far
        m += batch_size

        # Draw new samples from P(Q | E)
        new_samples = sample(spn, 
                             num_samples=batch_size,
                             evidence=evidence)
        
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
        new_probs = np.exp(ll_from_data(spn, unseen_samples) - p_evid)
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

    return [q_hat, p_hat]