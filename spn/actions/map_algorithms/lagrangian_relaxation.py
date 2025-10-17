"""Lagrangian Relaxation Methods"""
from typing import Dict, Tuple, List, cast

import numpy as np  # type: ignore

from spn.structs import Variable
from spn.node.base import SPN
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.utils.evidence import Evidence
from spn.actions.map_algorithms.bound_state import BoundState, get_default_bound_state


class LRDiag:
    """Helper Class to obtain diagnostic information about the Lagrangian
    Relaxation method. It allows for setting the learning parameters, obtain the
    converging values, and so on."""

    def __init__(
        self,
        spn: SPN,
        evidences: Evidence,
        bound_state: BoundState = None,
        max_iterations: int = 1,
        lambdas: List[Dict[Variable, Dict[int, float]]] = None,
    ):
        self.__spn = spn
        self.__evidences = evidences
        if bound_state is None:
            bound_state, _ = get_default_bound_state(self.__spn, self.__evidences)
        self.__bound_state = bound_state
        self.__values: List[float] = []
        self.__max_iterations = max_iterations
        if lambdas is None:
            self.__lambdas: List[Dict[Variable, Dict[int, float]]] = []
        else:
            self.__lambdas = lambdas

    @property
    def bound_state(self) -> BoundState:
        """The Bound State of this relaxation until now"""
        return self.__bound_state

    @property
    def values(self) -> List[float]:
        """List of values obtained in each iteration of the relaxation"""
        return self.__values

    def lagrangian_relaxation(self) -> Evidence:
        """Lagrangian relaxation algorithm auxiliar function to call the correct
        variant of the algorithm"""
        if len(self.__spn.scope()) == 1:
            return self.lagrangian_relaxation_scope_one(
                self.__spn, self.__evidences, self.__lambdas
            )
        if self.__spn.type == "product":
            return self.lagrangian_relaxation_product(
                cast(ProductNode, self.__spn), self.__evidences, self.__lambdas
            )
        return self.dual_decomp(
            cast(SumNode, self.__spn), self.__evidences, self.__lambdas
        )

    def lagrangian_relaxation_scope_one(
        self,
        spn: SPN,
        evidence: Evidence,
        lambs: List[Dict[Variable, Dict[int, float]]],
    ) -> Evidence:
        """Lagrangian relaxation for SPNs with scope of size 1. This is a brute
        force search on the space of solutions, taking O(n) time where n is the
        number of values the value X in the scope can be assigned."""
        variable = spn.scope()[0]
        best_value = -np.inf
        best_evidence = Evidence()
        for value in evidence[variable]:
            lambda_sum = sum([lamb[variable][value] for lamb in lambs])
            ev = Evidence({variable: [value]})
            new_value = spn.value(ev) * np.exp(-lambda_sum)
            if new_value > best_value:
                best_value = new_value
                best_evidence = ev
        return best_evidence

    def lagrangian_relaxation_product(
        self,
        spn: SPN,
        evidence: Evidence,
        lambs: List[Dict[Variable, Dict[int, float]]],
    ) -> Evidence:
        """Lagrangian Relaxation for Product Nodes, which decomposes the problem
        and merges the evidence found for each children"""
        final_evidence = Evidence()
        for child in spn.children:
            child_scope = child.scope()
            lambdas: List[Dict[Variable, Dict[int, float]]] = []
            new_evidence = Evidence()
            for lamb in lambs:
                for variable in lamb.keys():
                    if variable in child_scope:
                        lambdas += [lamb]
                        break
            for variable in child_scope:
                new_evidence[variable] = evidence[variable]

            subevidence = LRDiag(
                child, new_evidence, max_iterations=self.__max_iterations, lambdas=lambdas
            ).lagrangian_relaxation()
            final_evidence.merge(subevidence)

        return final_evidence

    def lambda_maximization(
        self,
        spn: SumNode,
        evidence: Evidence,
        child_maximizations: Dict[SPN, Evidence],
        lamb: Dict[SPN, Dict[Variable, Dict[int, float]]],
    ) -> Tuple[Evidence, float]:
        """Finds the maximum evidence based on the lambda values and the found best
        evidences for subproblems."""
        best_value = -np.inf
        best_evidence = Evidence()
        for subevidence in child_maximizations.values():
            total = 0.0
            for child, weight in zip(spn.children, spn.weights):
                sum_of_lambdas = 0.0
                child_evidence = child_maximizations[child]
                for variable in evidence.variables:
                    sum_of_lambdas += lamb[child][variable][subevidence[variable][0]]
                    sum_of_lambdas -= lamb[child][variable][child_evidence[variable][0]]
                total += weight * np.exp(sum_of_lambdas) * child.value(child_evidence)
            if total > best_value:
                best_evidence = subevidence
                best_value = total
        return best_evidence, best_value

    def dual_decomp(
        self,
        spn: SumNode,
        evidence: Evidence,
        lambs: List[Dict[Variable, Dict[int, float]]],
    ) -> Evidence:
        """Performs a dual decomposition on the MAP problem for the spn.
        Also performs the relaxation to solve the dual"""
        iteration = 0
        lamb = get_vector_in_dict_form(spn, evidence)

        while True:
            iteration += 1
            child_maximizations: Dict[SPN, Evidence] = {}

            for child in spn.children:
                child_maximizations[child] = LRDiag(
                    child,
                    evidence,
                    max_iterations=self.__max_iterations,
                    lambdas=lambs + [lamb[child]],
                ).lagrangian_relaxation()

            child_evidences_list = [
                child_maximizations[child] for child in spn.children
            ]

            evidence_maximization, max_value = self.lambda_maximization(
                cast(SumNode, spn), evidence, child_maximizations, lamb
            )

            maybe_new_lower = spn.value(evidence_maximization)
            if maybe_new_lower > self.__bound_state.lower:
                self.__bound_state.evidence = evidence_maximization
                self.__bound_state.lower = maybe_new_lower

            all_evidences = [evidence_maximization] + child_evidences_list

            if spn.root:
                if max_value < self.__bound_state.upper:
                    self.__bound_state.upper = max_value

                if iteration % 1 == 0 or iteration == 1:
                    self.__values += [max_value]

            if iteration >= self.__max_iterations and self.__bound_state.evidence is not None:
                return self.__bound_state.evidence

            if all_evidences.count(all_evidences[0]) == len(all_evidences):
                if spn.root:
                    self.__bound_state.evidence = evidence_maximization
                    self.__bound_state.lower = spn.value(self.__bound_state.evidence)
                return evidence_maximization

            grad = get_vector_in_dict_form(spn, evidence)
            norm_factor = 0.0

            for variable in evidence.variables:
                for child in spn.children:
                    current_child_evidence = child_maximizations[child]
                    assignment_child = current_child_evidence[variable][0]
                    assignment_individual = evidence_maximization[variable][0]

                    if assignment_child == assignment_individual:
                        continue

                    grad[child][variable][assignment_child] = -max_value
                    grad[child][variable][assignment_individual] = max_value

                    norm_factor += 2 * (max_value ** 2)

            rate = (
                2.0
                * ((self.__bound_state.upper * 1.05) - self.__bound_state.lower)
                / norm_factor
            )

            for variable in evidence.variables:
                for child in spn.children:
                    current_child_evidence = child_maximizations[child]
                    assignment_child = current_child_evidence[variable][0]
                    assignment_individual = evidence_maximization[variable][0]

                    lamb[child][variable][assignment_child] -= (
                        rate * grad[child][variable][assignment_child]
                    )
                    lamb[child][variable][assignment_individual] -= (
                        rate * grad[child][variable][assignment_individual]
                    )


def get_vector_in_dict_form(
    spn: SPN, evidence: Evidence
) -> Dict[SPN, Dict[Variable, Dict[int, float]]]:
    """Returns a 3-dimensional vector that can be indexed by an SPN, a variable, and a
    value from the possible values for the variable. The dictionary format makes it
    easier to read and to index its values"""
    dictionary: Dict[SPN, Dict[Variable, Dict[int, float]]] = {}
    for child in spn.children:
        dictionary[child] = {}
        for variable in evidence.variables:
            dictionary[child][variable] = {}
            for value in evidence[variable]:
                dictionary[child][variable][value] = 0.0
    return dictionary


def spn_value_with_lambdas(
    spn: SPN, evidence: Evidence, lambs: List[Dict[Variable, Dict[int, float]]]
) -> float:
    """Returns the value of the SPN evaluated with the evidence, penalized by the
    lambda values. This is used to calculate the value of the subproblems in the
    relaxation step."""
    lambda_sum = 0.0
    for lamb in lambs:
        for variable, values in evidence.items():
            lambda_sum += lamb[variable][values[0]]
    return spn.value(evidence) * np.exp(-lambda_sum)
