from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample_multiproc
from spn.actions.likelihood import likelihood_multiproc
from spn.actions.condition import condition_spn 
from spn.io.file import to_file, from_file 
import numpy as np
from spn.structs import Variable
from pathlib import Path
import tempfile
import time
import os


def pac_map_budget(
        spn: SPN, 
        spn_path: Path,
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        m: int = 1000,
        n_jobs: int = -1,
        warm_start_cands: List[Evidence] = None,
        warm_start_probs: List[float] = None
        ) -> Tuple[Evidence, float]:
    
    def sample_evid_to_tuple(evid):
        """Convert sample in Evidence() dictionary format to hashable tuple 
        (assumes 1 val per variable)."""
        return tuple(sorted((var.id, vals[0]) for var, vals in evid.items()))
    
    # Validate warm start inputs
    if warm_start_cands:
        assert warm_start_probs, (
            "If you are providing warm-start candidates, "
            "you need to provide their probabilities too"
        )
        assert len(warm_start_cands) == len(warm_start_probs), (
            "The list of warm start candidates has to be the same length "
            "as the list of their probabilities"
        )
    
    # Condition SPN on evidence once at the start
    if evidence and len(evidence) > 0:
        conditioned_spn = condition_spn(spn, evidence, marginalized)
        timestamp = int(time.time())
        conditioned_spn_path = spn_path.parent / f"{spn_path.stem}_conditioned_{timestamp}_{os.getpid()}.spn"
        to_file(conditioned_spn, conditioned_spn_path)
        working_path = conditioned_spn_path
        sampling_evidence = None
    else:
        conditioned_spn_path = None
        working_path = spn_path
        sampling_evidence = None
    
    # Initialize with warm start if you've given one
    candidate_list = list(warm_start_cands) if warm_start_cands else []
    probs = list(warm_start_probs) if warm_start_probs else []
    seen_hashes = set()
    
    # Add warm_start candidates to the set of seen samples 
    if warm_start_cands:
        for cand in warm_start_cands:
            sample_hash = sample_evid_to_tuple(cand)  
            seen_hashes.add(sample_hash)
    
    # Initialize p_hat and q_hat
    if len(probs) > 0:
        p_hat = max(probs)
        q_hat_idx = np.argmax(probs)
        q_hat = candidate_list[q_hat_idx]
    else:
        p_hat = float('-inf')
        q_hat = None

    try:
        # Draw new samples from P(Q | E) using conditioned SPN
        new_samples = sample_multiproc(working_path, 
                                        num_samples=m,
                                        evidence=sampling_evidence,  # None if conditioned
                                        n_jobs=n_jobs,
                                        marginalized=marginalized)
        
        # Add samples that haven't been seen before to candidate_list 
        unseen_samples = []
        for sample_dict in new_samples:
            filtered_sample = Evidence({var: vals for var, vals in sample_dict.items() 
                                            if var not in marginalized})
            sample_hash = sample_evid_to_tuple(filtered_sample)
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unseen_samples.append(filtered_sample)
                candidate_list.append(filtered_sample)
        
        # Compute P(Q | E) for new, unseen samples
        new_probs = np.exp(
            likelihood_multiproc(working_path, unseen_samples, n_jobs=n_jobs)
        )
        probs.extend(new_probs)

        # Check if you need to update q_hat and p_hat
        if max(probs) > p_hat:
            p_hat = max(probs)
            q_hat_idx = np.argmax(probs)
            q_hat = candidate_list[q_hat_idx]
        
        # Calculate pairs of values for epsilon and delta
        epsilon = np.linspace(0, 1 - p_hat - 1e-6, 200)
        delta = (1 - p_hat / (1 - epsilon)) ** M

    finally:
        # Cleanup: delete temporary conditioned SPN file
        if conditioned_spn_path is not None and conditioned_spn_path.exists():
            conditioned_spn_path.unlink()

    return [q_hat, p_hat, epsilon, delta]