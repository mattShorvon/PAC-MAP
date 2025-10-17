from typing import Tuple, cast, List, Dict, Optional
import operator
from functools import reduce
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.structs import Variable
from spn.node.indicator import Indicator
from spn.node.product import ProductNode
from spn.node.sum import SumNode
from spn.actions.map_algorithms.naive import naive_with_evidence


SubMapResult = Dict[SPN, Dict[Evidence, Tuple[Evidence, float]]]
HybridMaxProductResult = Tuple[Optional[Evidence], Optional[float], Optional[float]]

MAP_PROBLEM_LIMIT = 10
UPPER_BOUND_NODE_LIMIT = 10


def hybrid_max_product_indicator(
    spn: Indicator, evidence: Evidence
) -> HybridMaxProductResult:
    if spn.assignment in evidence[spn.variable]:
        return (Evidence({spn.variable: [spn.assignment]}), 1.0, 1.0)
    return None, None, 0.0


def hybrid_max_product_sum(
    spn: SumNode,
    evidence: Evidence,
    submapresult: SubMapResult,
    check_for_calculation: bool = True,
) -> HybridMaxProductResult:
    results: List[Tuple[Optional[Evidence], float]] = []
    upper_bound = None
    should_calculate = check_for_calculation and (
        evidence.map_problem_size() <= MAP_PROBLEM_LIMIT
        and spn.nodes() <= UPPER_BOUND_NODE_LIMIT
    )
    if should_calculate:
        upper_bound = 0.0
        check_for_calculation = False
        for child, weight in zip(spn.children, spn.weights):
            upper_bound_pair = None
            if child in submapresult:
                maps_for_child = submapresult[child]
                for macro_evidence, result in maps_for_child.items():
                    if macro_evidence.has_subevidence(evidence):
                        upper_bound_pair = result
                        break
            if upper_bound_pair is None:
                upper_bound_pair = naive_with_evidence(child, evidence)
                submapresult[child] = {evidence: upper_bound_pair}
            upper_bound += upper_bound_pair[1] * weight
    for child, weight in zip(spn.children, spn.weights):
        map_result = hybrid_max_product(
            child, evidence, submapresult, check_for_calculation
        )
        value = weight * map_result[1] if map_result[1] is not None else 0.0
        results.append((map_result[0], value))
        if map_result[2] is not None and not should_calculate:
            if upper_bound is None:
                upper_bound = map_result[2] * weight
            elif upper_bound is not None:
                upper_bound += map_result[2] * weight
    max_ev, max_value = max(results, key=lambda x: x[1])
    return max_ev, max_value, upper_bound


def hybrid_max_product_prod(
    spn: ProductNode,
    evidence: Evidence,
    submapresult: SubMapResult,
    check_for_calculation: bool = True,
) -> HybridMaxProductResult:
    new_evidence = Evidence()
    value = 1.0
    upper_bound = None
    upper_bounds = []
    for child in spn.children:
        child_evidence = Evidence(
            {var: values for var, values in evidence.items() if var in child.scope()}
        )
        result = hybrid_max_product(
            child, child_evidence, submapresult, check_for_calculation
        )
        if result[0] is not None:
            new_evidence.merge(result[0])
        if result[1] is not None:
            value *= result[1]
        upper_bounds.append(result[2])
    if upper_bounds[0] is not None:
        upper_bound = reduce(operator.mul, upper_bounds)
    return new_evidence, value, upper_bound


def hybrid_max_product(
    spn: SPN,
    evidence: Evidence,
    submapresult: SubMapResult,
    check_for_calculation: bool = True,
) -> HybridMaxProductResult:
    if spn.type == "leaf":
        return hybrid_max_product_indicator(cast(Indicator, spn), evidence)
    if spn.type == "sum":
        return hybrid_max_product_sum(
            cast(SumNode, spn), evidence, submapresult, check_for_calculation
        )
    # Product
    return hybrid_max_product_prod(
        cast(ProductNode, spn), evidence, submapresult, check_for_calculation
    )
