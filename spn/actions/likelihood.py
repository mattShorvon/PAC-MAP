"""Likelihood: Action for calculating Likelihood from an SPN"""

from typing import List
import logging
import math
from maua_et_al_experiments.database import DB
from spn.actions.base import Action
from spn.utils.evidence import Evidence
from spn.node.base import SPN
from joblib import Parallel, delayed


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


def ll_from_data(spn: SPN, evidences: List[Evidence]) -> float:
    """Returns the log-likelihood from a list of data instances"""
    results = Parallel(n_jobs=10)(
        delayed(spn.log_value)(evidence) for evidence in evidences
    )
    # return math.fsum(results) 
    return results # originally this returned the sum of results, 
                   # but we want the results themselves
