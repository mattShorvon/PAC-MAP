from typing import Tuple, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.likelihood import likelihood_multiproc, ll_from_data
from spn.io.file import to_file, from_file 
import numpy as np
from spn.structs import Variable
from pathlib import Path
import tempfile
import time
import os
import copy

# Naive baseline for MAP estimation that optimises each query variable 
# independently
def independent_map(
        spn: SPN = None,
        spn_path: Path = None,
        evidence: Evidence = None,
        p_evid: float = 0,
        marginalised: List[Variable] = None,
        n_jobs: int = -2
):
    # Check if evidence/marginalised are empty
    if not evidence:
        evidence = Evidence()
    if not marginalised:
        marginalised = Evidence()

    # Get the query variables
    q_vars = [
        var for var in spn.scope() if var not in evidence and var not in marginalised
    ]

    # Iterate through each variable
    q_hat = copy.deepcopy(evidence)
    for var in q_vars:
        # Create the list of queries to send to likelihood()
        queries = []
        for val in range(var.n_categories):
            evid = copy.deepcopy(evidence)
            evid.merge(Evidence({var: [val]}))
            queries.append(evid)
        
        # Send the queries off to likelihood()
        probs = []
        for query in queries:
            log_prob = spn.log_value(query)
            probs.append(np.exp(log_prob - p_evid))

        # Find the highest probability assignment for this variable
        p_hat_var = max(probs)
        var_hat_idx = np.argmax(probs)
        var_hat = queries[var_hat_idx][var]
        q_hat.merge(Evidence({var: var_hat}))
    
    # Calculate p_hat and return
    p_hat = np.exp(spn.log_value(q_hat) - p_evid)
    return [q_hat, p_hat]