"""Bag: Action for applying bagging to learn SPNs"""

import logging
from maua_et_al_experiments.database import DB
from spn.learn import gens
from spn.actions.base import Action
from spn.actions.em import em_with_restarts
from spn.node.sum import SumNode
from spn.node.base import SPN
from spn.data.parsed_data import ParsedData
from joblib import Parallel, delayed


class Bag(Action):
    necessary_params = [
        "training-dataset",
        "test-dataset",
        "kclusters",
        "pval",
        "name",
        "n-bags",
        "iterations",
        "laplacian",
        "epsilon",
        "restarts",
    ]
    key = "Bag"

    def execute(self):
        """Uses the dataset to learn an SPN and bag n models into a larger SPN"""
        logging.info("Executing Bagging action")
        training_data = DB.get(self.params["training-dataset"])
        test_data = DB.get(self.params["test-dataset"])
        kclusters = self.params["kclusters"]
        pval = self.params["pval"]
        n_bags = self.params["n-bags"]

        iterations = self.params["iterations"]
        laplacian = self.params["laplacian"]
        epsilon = self.params["epsilon"]
        restarts = self.params["restarts"]

        spn = do_bagging_with_em(
            training_data,
            test_data,
            kclusters,
            pval,
            n_bags,
            iterations,
            laplacian,
            epsilon,
            restarts,
        )

        print("SPN learned:", spn.vars(), "vars and", spn.arcs(), "arcs")
        DB.store(self.params["name"], spn)


def do_bagging_with_em(
    training_data: ParsedData,
    test_data: ParsedData,
    kclusters: int,
    pval: float,
    n_bags: int,
    em_iterations: int,
    em_laplacian: float,
    em_epsilon: float,
    em_restarts: int,
) -> SPN:
    root = SumNode()
    uniform_weight = 1.0 / float(n_bags)
    em_spns = Parallel(n_jobs=12)(
        delayed(learn_spn_with_em)(
            training_data,
            test_data,
            kclusters,
            pval,
            em_iterations,
            em_laplacian,
            em_epsilon,
            em_restarts,
        )
        for _ in range(n_bags)
    )
    for em_spn in em_spns:
        root.add_child(em_spn, uniform_weight)
    root.root = True
    return root


def learn_spn_with_em(
    training_data: ParsedData,
    test_data: ParsedData,
    kclusters: int,
    pval: float,
    iterations: int,
    laplacian: float,
    epsilon: float,
    restarts: int,
) -> SPN:
    new_data = training_data.uniform_sample()
    spn = gens(new_data.scope, new_data.data, kclusters, pval, False)
    fitted_spn = em_with_restarts(
        spn, training_data, test_data, iterations, laplacian, epsilon, restarts
    )
    return fitted_spn
