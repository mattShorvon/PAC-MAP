from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.sample import sample_multiproc
from spn.actions.likelihood import likelihood_multiproc
from spn.actions.condition import condition_spn  # ← NEW
from spn.io.file import to_file, from_file  # ← NEW
import numpy as np
from spn.structs import Variable
from pathlib import Path
import tempfile


def pac_map(
        spn: SPN, 
        spn_path: Path,
        evidence: Evidence, 
        marginalized: List[Variable] = [],
        batch_size: int = 10,
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
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.spn', delete=False) as f:
            conditioned_spn_path = Path(f.name)
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
    
    def sample_evid_to_tuple(evid):
        """Convert sample in Evidence() dictionary format to hashable tuple 
        (assumes 1 val per variable)."""
        return tuple(sorted((var.id, vals[0]) for var, vals in evid.items())) 

    try:
        while m < M:
            # Update number of samples taken so far
            m += batch_size

            # Draw new samples from P(Q | E) using conditioned SPN
            new_samples = sample_multiproc(working_path, 
                                           num_samples=batch_size,
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

            # Check if you need to update the best candidate
            if max(probs) > p_hat:
                p_hat = max(probs)
                q_hat_idx = np.argmax(probs)
                q_hat = candidate_list[q_hat_idx]

            # Check if you can issue the PAC certificate
            p_tick = 1 - sum(probs)
            if p_tick <= p_hat/(1 - err_tol):
                M = 0
            else:
                M = np.log(1/fail_prob)/(p_hat/(1 - err_tol)) # Currently using approximate calculation
            
            # Check if the sample_cap has been hit
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