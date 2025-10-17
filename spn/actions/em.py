"""EM: Action for running the cccp algorithm on learned SPNs"""

import sys
import logging
import copy
from typing import Dict, cast, List
import math
import numpy
from database import DB
from spn.data.partitioned_data import ParsedData
from spn.node.base import SPN
from spn.actions.base import Action
from spn.actions.likelihood import ll_from_data
from spn.node.sum import SumNode
from spn.node.indicator import Indicator
from joblib import Parallel, delayed


class EM(Action):
    necessary_params = [
        "spn",
        "training-dataset",
        "test-dataset",
        "iterations",
        "laplacian",
        "epsilon",
        "name",
        "restarts",
    ]
    key = "EM"

    def execute(self):
        """Uses the training and validation datasets to run the
        Expectation-Maximization algorithm on the SPN"""
        logging.info("Executing Expectation-Maximization action")
        spn = DB.get(self.params["spn"])
        training_data = DB.get(self.params["training-dataset"])
        test_data = DB.get(self.params["test-dataset"])
        if training_data is None or test_data is None:
            logging.error(
                f"No data for SPN {self.params['spn']}. Cancelling EM action..."
            )
            return
        final_spn = em_with_restarts(
            spn,
            training_data,
            test_data,
            self.params["iterations"],
            self.params["laplacian"],
            self.params["epsilon"],
            self.params["restarts"],
        )
        DB.store(self.params["name"], final_spn)


def em_with_restarts(
    spn: SPN,
    training_data: ParsedData,
    test_data: ParsedData,
    iterations: int,
    lap_lambda: float,
    stop_epsilon: float,
    restarts: int,
) -> SPN:
    logging.info("Starting EM with restarts")
    final_spn = copy.deepcopy(spn)
    logging.info("SPN copied")
    best_ll_train = ll_from_data(final_spn, training_data.generate_evidences())
    best_ll_test = ll_from_data(final_spn, test_data.generate_evidences())
    for _ in range(restarts):
        spn_copy = copy.deepcopy(spn)
        new_spn = do_em(
            spn_copy, training_data, test_data, iterations, lap_lambda, stop_epsilon
        )
        new_spn_ll_train = ll_from_data(new_spn, training_data.generate_evidences())
        new_spn_ll_test = ll_from_data(new_spn, test_data.generate_evidences())
        print(new_spn_ll_train)
        print(new_spn_ll_test)
        print('-' * 80)
        if new_spn_ll_train + new_spn_ll_test >= best_ll_train + best_ll_test:
            best_ll_train = new_spn_ll_train
            best_ll_test = new_spn_ll_test
            final_spn = new_spn
    return final_spn


def do_em(
    spn: SPN,
    training_data: ParsedData,
    test_data: ParsedData,
    iterations: int,
    lap_lambda: float,
    stop_epsilon: float,
) -> SPN:
    """Runs the Expectation-Maximization algorithm with passed data"""
    sum_nodes = []
    sufficient_stats: Dict[SPN, List[float]] = {}
    optimization: Dict[SPN, List[float]] = {}
    for node in spn.topological_order():
        if node.type == "sum":
            node = cast(SumNode, node)
            len_children = len(node.children)
            sum_nodes.append(node)
            sufficient_stats[node] = [0.0] * len_children
            optimization[node] = [0.0] * len_children
        elif node.type == 'leaf':
            node = cast(Indicator, node)

    train_funcs: List[float] = []
    valid_funcs: List[float] = []
    optimal_valid_prob = -math.inf

    epsilon = sys.float_info.epsilon

    training_evidences = training_data.generate_evidences()
    # Using test data as validation
    validation_evidences = test_data.generate_evidences()

    # Always normalize
    #for sum_node in sum_nodes:
    #    sum_node.normalize_weights()

    logging.info("#iteration,train-lld,valid-lld")

    for t in range(iterations):
        train_logprobs = 0.0
        valid_logprobs = 0.0

        # Clean previous statistics
        for key in sufficient_stats:
            current_value = sufficient_stats[key]
            sufficient_stats[key] = [0.0] * len(current_value)

        for evidence in training_evidences:
            evaluation = spn.eval(evidence)
            eval_values = {
                node: evaluation[index]
                for index, node in enumerate(reversed(spn.topological_order()))
            }
            derivatives = spn.derivatives(evidence, evaluation)
            derivative_values = {
                node: derivatives[index]
                for index, node in enumerate(reversed(spn.topological_order()))
            }
            root_value = evaluation[-1]
            train_logprobs += root_value
            for node in sum_nodes:
                for child_index, (child, weight) in enumerate(
                    zip(node.children, node.weights)
                ):
                    sufficient_stats[node][child_index] += (
                        weight
                        * derivative_values[node]
                        * eval_values[child]
                        / root_value
                    )
        for evidence in validation_evidences:
            valid_logprobs += spn.log_value(evidence)

        train_logprobs /= len(training_evidences)
        valid_logprobs /= len(validation_evidences)

        # Store statistics
        train_funcs += [train_logprobs]
        valid_funcs += [valid_logprobs]

        if valid_logprobs > optimal_valid_prob:
            optimal_valid_prob = valid_logprobs
            for node in sum_nodes:
                for child_index, weight in enumerate(node.weights):
                    optimization[node][child_index] = weight

        logging.info(f"{t},{train_funcs[t]},{valid_funcs[t]}")

        for node in sum_nodes:
            ssz = 0.0
            for child_index in range(len(node.children)):
                ssz += sufficient_stats[node][child_index] + epsilon
            # Weight udpate
            for child_index, child in enumerate(node.children):
                node.set_weight_at(
                    (sufficient_stats[node][child_index] + epsilon) / ssz,
                    child_index,
                )

        if t > 0 and train_funcs[t] - train_funcs[t - 1] < stop_epsilon:
            break
    # Restore the optimal model weight parameter encountered during optimization
    for node in sum_nodes:
        for child_index, child in enumerate(node.children):
            node.set_weight_at(optimization[node][child_index], child_index)
    return spn
