from typing import Callable, List, cast, Tuple, Optional, Dict
import math
from spn.utils.evidence import Evidence
from spn.node.base import SPN
from spn.structs import Variable
from spn.utils.evidence import Evidence

DerivativesCache = Optional[Dict[int, List[float]]]
BranchingHeuristic = Callable[
    [SPN, Evidence, Optional[List[Variable]], DerivativesCache],
    Optional[Tuple[Variable, List[int]]],
]


def size_of_second_value(x):
    return len(x[1])


def first_available_variable(
    _: SPN,
    evidence: Evidence,
    marginalized_variables: Optional[List[Variable]],
    __: DerivativesCache,
) -> Optional[Tuple[Variable, List[int]]]:
    for variable, values in evidence.items():
        if marginalized_variables is not None and variable in marginalized_variables:
            continue
        if len(values) > 1:
            return variable, values
    return None


def lowest_marginal(
    spn: SPN,
    evidence: Evidence,
    marginalized_variables: Optional[List[Variable]],
    derivatives: DerivativesCache,
) -> Optional[Tuple[Variable, List[int]]]:
    derivative_of_assignment = (
        derivatives
        if derivatives is not None
        else spn.derivative_of_assignment(evidence)
    )
    chosen_var_and_values = None
    lowest_value = 1.0
    for variable, values in evidence.items():
        if len(values) <= 1:
            continue
        sum_of_marginals = min(derivative_of_assignment[variable.id])
        if sum_of_marginals < lowest_value:
            chosen_var_and_values = (variable, values)
            lowest_value = sum_of_marginals
    return chosen_var_and_values


def largest_marginal(
    spn: SPN, evidence: Evidence, derivatives: DerivativesCache
) -> Optional[Tuple[Variable, List[int]]]:
    derivative_of_assignment = (
        derivatives
        if derivatives is not None
        else spn.derivative_of_assignment(evidence)
    )
    chosen_var_and_values = None
    largest_value = 0.0
    for variable, values in evidence.items():
        if len(values) <= 1:
            continue
        sum_of_marginals = max(derivative_of_assignment[variable.id])
        if sum_of_marginals > largest_value:
            chosen_var_and_values = (variable, values)
            largest_value = sum_of_marginals
    return chosen_var_and_values


def lowest_entropy(
    spn: SPN, evidence: Evidence, derivatives: DerivativesCache
) -> Optional[Tuple[Variable, List[int]]]:
    derivative_of_assignment = (
        derivatives
        if derivatives is not None
        else spn.derivative_of_assignment(evidence)
    )
    chosen_var_and_values = None
    lowest_value = 1.0
    for variable, values in evidence.items():
        if len(values) <= 1:
            continue
        log_base = len(values)
        entropy = -math.fsum(
            [
                value * math.log(value, log_base)
                for value in derivative_of_assignment[variable.id]
            ]
        )
        if entropy < lowest_value:
            chosen_var_and_values = (variable, values)
            lowest_value = entropy
    return chosen_var_and_values


def largest_entropy(
    spn: SPN, evidence: Evidence, derivatives: DerivativesCache
) -> Optional[Tuple[Variable, List[int]]]:
    derivative_of_assignment = (
        derivatives
        if derivatives is not None
        else spn.derivative_of_assignment(evidence)
    )
    chosen_var_and_values = None
    largest_value = 0.0
    for variable, values in evidence.items():
        if len(values) <= 1:
            continue
        log_base = len(values)
        entropy = -math.fsum(
            [
                value * math.log(value, log_base)
                for value in derivative_of_assignment[variable.id]
            ]
        )
        if entropy > largest_value:
            chosen_var_and_values = (variable, values)
            largest_value = entropy
    return chosen_var_and_values


def ordering(
    spn: SPN, evidence: Evidence, derivatives: DerivativesCache
) -> Optional[Tuple[Variable, List[int]]]:
    item_to_choose = min(
        [(var, value) for var, value in evidence.items() if len(value) > 1],
        key=size_of_second_value,
    )
    if item_to_choose:
        derivatives = (
            derivatives
            if derivatives is not None
            else spn.derivative_of_assignment(evidence)
        )
        derivatives = spn.derivative_of_assignment(evidence)
        chosen_variable, values_for_chosen_variable = item_to_choose
        values_for_chosen_variable = sorted(
            values_for_chosen_variable,
            key=lambda value: derivatives[chosen_variable.id][value],
            reverse=True,
        )
        return (chosen_variable, values_for_chosen_variable)
    return None
