"""New Implementation of Branch and Bound methods for MAP in SPNs"""

import copy
from typing import cast, Dict, Optional, Tuple, List
from collections import deque
import numpy as np
import math
import time

from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.utils.evidence import Evidence
from spn.actions.map_algorithms.bound_state import BoundState
from spn.actions.map_algorithms.hybrid_max_product import (
    hybrid_max_product,
    SubMapResult,
)
from spn.actions.map_algorithms.max_product import max_product_with_evidence
from spn.actions.map_algorithms.max_search import forward_checking
from spn.structs import Variable


def do_branch_and_bound(spn: SPN) -> Evidence:
    """Function to call branch and bound with all variables marginalized"""
    evidence = spn.all_marginalized()
    diagnostic = BBDiag(spn, evidence)
    diagnostic.run()
    return diagnostic.bound_state.evidence


class BBDiag:
    """Diagnostic information for Branch and Bound to track changes in evidences and
    bounds"""

    def __init__(self, spn: SPN, evidences: Evidence):
        self.__spn = spn
        self.__evidences = evidences
        self.__bound_state = BoundState(0.0, 1.0, Evidence())
        self.__submapresult: SubMapResult = {}
        self.__size = 0
        self.__pruned = 0

    @property
    def bound_state(self) -> BoundState:
        return self.__bound_state

    @property
    def size(self) -> int:
        """Size is the number of nodes opened for this B&B"""
        return self.__size

    @property
    def pruned(self) -> int:
        """Pruned is the number of pruned nodes for this B&B"""
        return self.__pruned

    def set_bound_state(self, bound_state: BoundState):
        self.__bound_state = bound_state

    def update_bound_state(self, lower: float, upper: float, evidence: Evidence):
        """Sets the new bounds for this step of the branch and bound algorithm"""
        self.__bound_state = BoundState(lower, upper, evidence)

    def map_leaf(self):
        evidence = copy.deepcopy(self.__evidences)
        indicator = cast(Indicator, self.__spn)
        if indicator.assignment in evidence[indicator.variable]:
            evidence[self.__spn.variable] = [self.__spn.assignment]
        else:
            raise ValueError(
                f"The evidence restriction cannot be applied to this Indicator \
                with variable {indicator.variable} and assignment \
                {indicator.assignment}"
            )
        self.update_bound_state(1.0, 1.0, evidence)

    def map_product(self):
        results = []
        for child in self.__spn.children:
            child_evidence = Evidence()
            for variable in child.scope():
                child_evidence[variable] = self.__evidences[variable]
            diag = BBDiag(child, child_evidence)
            diag.run()
            results.append(diag)
        new_evidence = Evidence()
        for result in results:
            new_evidence.merge(result.bound_state.evidence)
        value = self.__spn.value(new_evidence)
        self.update_bound_state(value, value, new_evidence)

    def map_sum(self):
        if self.__evidences.all_set():
            value = self.__spn.value(self.__evidences)
            self.update_bound_state(value, value, self.__evidences)
            return
        nodes = deque()
        best_evidence = Evidence()
        lower_bound = 0.0
        var_to_branch, values_to_branch = branching(self.__evidences)
        for value_to_branch in values_to_branch:
            new_evidence = copy.deepcopy(self.__evidences)
            new_evidence[var_to_branch] = [value_to_branch]
            nodes.append(self.create_node_to_expand(new_evidence))
            self.__size += 1

        while nodes:
            local_lower, lower_bound_evidence, local_upper_bound, evidences_to_consider = (
                nodes.popleft()
            )

            if local_upper_bound < lower_bound:
                self.__pruned += 1
                continue
            if local_lower > lower_bound:
                print(f"{time.time()},{local_lower}")
                lower_bound = local_lower
                best_evidence = lower_bound_evidence
                indexes_to_remove = []
                for index, element in enumerate(nodes):
                    if element[2] < lower_bound:
                        indexes_to_remove.append(index - len(indexes_to_remove))
                for index in indexes_to_remove:
                    del nodes[index]
            if evidences_to_consider.all_set():
                value = self.__spn.value(evidences_to_consider)
                if value > lower_bound:
                    lower_bound = value
                    best_evidence = evidences_to_consider
            else:
                var_to_branch, values_to_branch = branching(evidences_to_consider)
                for value in values_to_branch:
                    new_evidence = copy.deepcopy(evidences_to_consider)
                    new_evidence[var_to_branch] = [value]
                    new_lower_bound, new_lower_bound_evidence, new_upper_bound, new_evidence = self.create_node_to_expand(
                        new_evidence
                    )
                    if new_upper_bound < lower_bound:
                        self.__pruned += 1
                        continue
                    nodes.append(
                        (
                            new_lower_bound,
                            new_lower_bound_evidence,
                            new_upper_bound,
                            new_evidence,
                        )
                    )
                    self.__size += 1
        self.update_bound_state(lower_bound, 1.0, best_evidence)

    def run(self):
        """Branch-and-Bound algorithm using Lagrangian Relaxation to obtain
        upper bounds"""
        if self.__spn.type == "leaf":
            self.map_leaf()
        elif self.__spn.type == "product":
            self.map_product()
        else:
            self.map_sum()

    def create_node_to_expand(
        self, evidence: Evidence
    ) -> Tuple[float, Evidence, float, Evidence]:
        max_prod_evidence, _, upper_bound = hybrid_max_product(
            self.__spn, evidence, self.__submapresult
        )
        lower_bound = self.__spn.value(max_prod_evidence)

        return lower_bound, max_prod_evidence, upper_bound, evidence


def branching(evidence: Evidence) -> Optional[Tuple[Variable, List[int]]]:
    var_and_values_to_branch = None
    max_values = 1
    for variable in evidence.variables:
        values = evidence[variable]
        n_values = len(values)
        if n_values > max_values:
            max_values = n_values
            var_and_values_to_branch = variable, values
    if var_and_values_to_branch is not None:
        return var_and_values_to_branch
    return None
