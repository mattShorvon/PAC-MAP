import copy
import signal
from typing import Callable, cast, Optional, Tuple, List
import operator
from functools import reduce
import math
from spn.utils.evidence import Evidence
from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.constant import Constant
from spn.actions.map_algorithms.anytime_info import AnytimeInfo
from spn.actions.map_algorithms.heuristics import (
    BranchingHeuristic,
    DerivativesCache,
    first_available_variable,
    size_of_second_value,
)
from spn.structs import Variable

MIN_VARIABLES_TO_ACTIVATE_STAGE = 5

CheckingFunction = Callable[
    [SPN, Evidence, float, Optional[List[Variable]]], Tuple[Evidence, DerivativesCache]
]


class TimeoutException(Exception):
    pass


def timeout_handler(_, __):
    raise TimeoutException("timeout")


def max_search(
    spn: SPN,
    checking_function: CheckingFunction,
    branching_heuristic: BranchingHeuristic = first_available_variable,
    time_limit: Optional[int] = None,
    marginalized_variables: Optional[List[Variable]] = None,
    evidence: Optional[Evidence] = None,
) -> Tuple[Evidence, AnytimeInfo]:
    """Max-Search is Mei's algorithm found in https://arxiv.org/abs/1708.04846. It
    works similarly to a branch-and-bound with different algorithms for checking
    the current bound, such as marginal checking and forward checking.

    This is the general method. It can be specialized through the checking algorithm"""
    anytime_info = AnytimeInfo()
    partial_evidences = spn.all_marginalized()
    initial_evidence = Evidence({var: [0] for var in spn.scope()})
    if evidence is not None:
        for var, values in evidence.items():
            if len(values) == 1:
                initial_evidence[var] = values
            partial_evidences[var] = copy.deepcopy(values)
    if marginalized_variables is not None:
        for variable in marginalized_variables:
            initial_evidence[variable] = list(range(variable.n_categories))
    initial_value = spn.value(initial_evidence)
    anytime_info.new_lower_bound(initial_value, initial_evidence)
    if time_limit is not None:
        evidence_found = Evidence()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)
        try:
            search(
                spn,
                partial_evidences,
                initial_evidence,
                initial_value,
                checking_function,
                branching_heuristic,
                anytime_info,
                marginalized_variables,
            )
            evidence_found = anytime_info.best_evidence()
            anytime_info.finish()
        except TimeoutException:
            evidence_found = anytime_info.best_evidence()
        finally:
            signal.alarm(0)
        return evidence_found, anytime_info
    return (
        search(
            spn,
            partial_evidences,
            initial_evidence,
            initial_value,
            checking_function,
            branching_heuristic,
            anytime_info,
            marginalized_variables,
        ),
        anytime_info,
    )


def max_search_with_ordering_and_staging(
    spn: SPN, checking_function: CheckingFunction
) -> Evidence:
    """Max-Search is Mei's algorithm found in https://arxiv.org/abs/1708.04846. It
    works similarly to a branch-and-bound with different algorithms for checking
    the current bound, such as marginal checking and forward checking.

    This is the general method. It can be specialized through the checking algorithm"""
    partial_evidences = spn.all_marginalized()
    initial_evidence = Evidence({var: [0] for var in spn.scope()})
    initial_value = spn.value(initial_evidence)
    return search_with_ordering_and_staging(
        spn, partial_evidences, initial_evidence, initial_value, checking_function
    )


def search(
    spn: SPN,
    partial_evidences: Evidence,
    best_evidence: Evidence,
    best_value: float,
    checking_function: CheckingFunction,
    branching_heuristic: BranchingHeuristic,
    anytime_info: Optional[AnytimeInfo] = None,
    marginalized_variables: Optional[List[Variable]] = None,
    derivatives_cache: DerivativesCache = None,
) -> Evidence:
    """
    The MAX-Search algorithm by Mei et al (2018).
    """
    # Select a variable that still needs a value determined
    if marginalized_variables is not None:
        non_marginalized_variables = [
            var for var in spn.scope() if var not in marginalized_variables
        ]
        if all(
            [len(partial_evidences[var]) == 1 for var in non_marginalized_variables]
        ):
            return partial_evidences
    if partial_evidences.all_set():
        # All variables are determined
        return partial_evidences
    chosen_variable_and_values = branching_heuristic(
        spn, partial_evidences, marginalized_variables, derivatives_cache
    )
    if chosen_variable_and_values is not None:
        chosen_variable, values_for_chosen_variable = chosen_variable_and_values
        for possible_value in values_for_chosen_variable:
            evidences_to_consider = copy.deepcopy(partial_evidences)
            evidences_to_consider[chosen_variable] = [possible_value]
            pruned_evidences, derivatives_cache = checking_function(
                spn, evidences_to_consider, best_value, marginalized_variables
            )
            # Check for still more evidences to traverse
            if pruned_evidences:
                returned_evidence = search(
                    spn,
                    pruned_evidences,
                    best_evidence,
                    best_value,
                    checking_function,
                    branching_heuristic,
                    anytime_info,
                    marginalized_variables,
                    derivatives_cache,
                )
                if returned_evidence != best_evidence:
                    best_evidence = returned_evidence
                    best_value = spn.value(best_evidence)
                    if anytime_info is not None:
                        anytime_info.new_lower_bound(best_value, best_evidence)
    return best_evidence


def search_with_ordering_and_staging(
    spn: SPN,
    partial_evidences: Evidence,
    best_evidence: Evidence,
    best_value: float,
    checking_function: CheckingFunction,
) -> Evidence:
    """
    The MAX-Search algorithm by Mei et al (2018).
    """
    # Select a variable that still needs a value determined
    if partial_evidences.all_set():
        # All variables are determined
        return partial_evidences
    set_variables = []
    unset_variables = []
    for var, values in partial_evidences.items():
        if len(values) == 1:
            set_variables.append((var, values))
        else:
            unset_variables.append((var, values))
    if len(set_variables) >= MIN_VARIABLES_TO_ACTIVATE_STAGE:
        # This is necessary for the current staging operation, which is generic to
        # the point of allowing marginalized evidences to be computed and
        # transform the SPN into a constant. For example, a completely marginalized
        # evidence would transform the SPN into a Constant node with value 1.
        # We create a new evidence, without some variables, in order to avoid this
        single_evidence = Evidence(
            {variable: values for variable, values in set_variables}
        )
        spn = stage(spn, single_evidence)
        spn.fix_topological_order()
    item_to_choose = min(unset_variables, key=size_of_second_value)
    if item_to_choose:
        chosen_variable, values_for_chosen_variable = item_to_choose
        derivatives = spn.derivative_of_assignment(partial_evidences)
        values_for_chosen_variable = sorted(
            values_for_chosen_variable,
            key=lambda value: derivatives[chosen_variable.id][value],
            reverse=True,
        )
        for value in values_for_chosen_variable:
            new_evidence = copy.deepcopy(partial_evidences)
            new_evidence[chosen_variable] = [value]
            pruned_evidences = checking_function(spn, new_evidence, best_value)
            # Check for still more evidences to traverse
            if pruned_evidences:
                returned_evidence = search_with_ordering_and_staging(
                    spn, pruned_evidences, best_evidence, best_value, checking_function
                )
                if returned_evidence != best_evidence:
                    best_evidence = returned_evidence
                    best_value = spn.value(best_evidence)
    return best_evidence


def marginal_checking(
    spn: SPN,
    partial_evidences: Evidence,
    best_value: float,
    _: Optional[List[Variable]],
) -> Tuple[Evidence, DerivativesCache]:
    """Marginal Checking. Sums all the values of the evidences in the partial
    evidences set"""
    if spn.value(partial_evidences) > best_value:
        return partial_evidences, None
    return Evidence(), None


def forward_checking(
    spn: SPN,
    partial_evidences: Evidence,
    best_value: float,
    marginalized_variables: Optional[List[Variable]],
) -> Tuple[Evidence, DerivativesCache]:
    """Forward Checking: Obtain the derivative of the current possible values for the
    SPN and prune the Assignments with a lower value than the current one"""
    evidences_changed = True
    derivatives = None
    while evidences_changed:
        evidences_changed = False
        derivatives = spn.derivative_of_assignment(partial_evidences)
        new_evidences = copy.deepcopy(partial_evidences)
        for variable, possible_assignments in partial_evidences.items():
            if (
                marginalized_variables is not None
                and variable in marginalized_variables
            ):
                continue
            for assignment in possible_assignments:
                if (
                    not math.isclose(best_value, derivatives[variable.id][assignment])
                    and best_value > derivatives[variable.id][assignment]
                ):
                    if len(new_evidences[variable]) == 1:
                        # This avoids an evidence with one variable without assignment.
                        # Such a case should be handled as no assignment to any var
                        return Evidence(), derivatives
                    new_evidences[variable].remove(assignment)
                    evidences_changed = True
        partial_evidences = new_evidences
    return partial_evidences, derivatives


def merge_two_evidences(ev1: Evidence, ev2: Evidence):
    ev1.merge(ev2)
    return ev1


def stage(spn: SPN, evidence: Evidence) -> SPN:
    """Transforms nodes into constant nodes if their values can be determined given
    the evidence, shrinking the size of the SPN"""
    if isinstance(spn, Constant):
        return spn
    if isinstance(spn, Indicator):
        indicator = cast(Indicator, spn)
        if spn.variable in evidence:
            return Constant(
                1 if indicator.assignment in evidence[indicator.variable] else 0,
                [indicator.variable],
                Evidence({indicator.variable: [indicator.assignment]}),
            )
    else:
        staged_children = [stage(child, evidence) for child in spn.children]
        if all([isinstance(child, Constant) for child in staged_children]):
            assignments = reduce(
                merge_two_evidences, [child.assignments() for child in staged_children]
            )
            if spn.type == "product":
                return Constant(
                    reduce(
                        operator.mul,
                        [child.value(evidence) for child in staged_children],
                    ),
                    spn.scope(),
                    assignments,
                )
            if spn.type == "sum":
                return Constant(
                    sum(
                        [
                            child.value(evidence) * weight
                            for child, weight in zip(staged_children, spn.weights)
                        ]
                    ),
                    spn.scope(),
                    assignments,
                )
        if spn.type == "sum":
            new_spn = SumNode()
            new_spn.weights = spn.weights
        elif spn.type == "product":
            new_spn = ProductNode()
        new_spn.children = staged_children
        return new_spn
    return spn
