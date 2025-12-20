"""Likelihood: Action for calculating Likelihood from an SPN"""

from typing import List
import logging
import math
from maua_et_al_experiments.database import DB
from spn.actions.base import Action
from spn.utils.evidence import Evidence
from spn.node.base import SPN
from joblib import Parallel, delayed
from pathlib import Path
from spn.io.file import from_file
import os
import numpy as np

class Likelihood(Action):
    necessary_params = ["spn", "training-dataset", "test-dataset"]
    key = "LL"

    def execute(self):
        """Uses the dataset associated with the SPN to calculate Log Likelihood"""
        logging.info("Executing Log-Likelihood action")
        spn = DB.get(self.params["spn"])
        training_data = DB.get(self.params["training-dataset"])
        test_data = DB.get(self.params["test-dataset"])
        if training_data is None or test_data is None:
            logging.error(
                f"No data for SPN {self.params['spn']}. Cancelling EM action..."
            )
            return
        train_evidences = training_data.generate_evidences()
        test_evidences = test_data.generate_evidences()
        train_ll = ll_from_data(spn, train_evidences)
        test_ll = ll_from_data(spn, test_evidences)
        print(f"For SPN {self.params['spn']}:")
        print(f"  TrainLL = {train_ll}")
        print(f"  TestLL = {test_ll}")

def _likelihood(spn: SPN = None, evidences: List[Evidence] = None):
    results = []
    for evid in evidences:
        ll = spn.log_value(evid)
        results.append(ll)
    return results

def _likelihood_worker(spn_path: Path = None, 
                       evidences: List[Evidence] = None) -> float:
    spn = from_file(spn_path)
    return _likelihood(spn, evidences)

def likelihood_multiproc(spn_path: Path = None, 
                         evidences: List[Evidence] = None,
                         n_jobs: int = -1) -> List:
    if n_jobs < 0:
        n_cores = os.cpu_count() + 1 + n_jobs
    else:
        n_cores = n_jobs
    n_workers = max(1, n_cores)

    evidence_arrays = np.array_split(evidences, n_workers)
    evidence_batches = [array.to_list() for array in evidence_arrays]

    results = Parallel(n_jobs=n_workers)(
        delayed(_likelihood_worker)(spn_path, evid_batch) 
        for evid_batch in evidence_batches
    )
    return results

def ll_from_data(spn: SPN, evidences: List[Evidence]) -> List:
    """Returns the log-likelihood from a list of data instances"""
    results = Parallel(n_jobs=10, backend='threading')(
        delayed(spn.log_value)(evidence) for evidence in evidences
    )
    # return math.fsum(results) 
    return results # originally this returned the sum of results, 
                   # but we want the results themselves
