from typing import Tuple, Optional, cast, List
from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.node.gaussian import GaussianNode
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.utils.evidence import Evidence
from spn.structs import Variable


def do_max_product(spn: SPN) -> Evidence:
    """Caller for max_product to adequate returned value to be only
    the Evidence"""
    # TODO: Change the recursive max_product function to return only the
    # evidence
    return max_product(spn)[0]


def max_product(spn: SPN) -> Tuple[Evidence, float]:
    """Max-product algorithm based on 'Sum-product networks: A new deep architecture
    Inference in Sum-Product Networks'"""
    if spn.type == "leaf":
        spn = cast(Indicator, spn)
        return Evidence({spn.variable: [spn.assignment]}), 1.0
    elif spn.type == "gaussian":
        evidence = Evidence({spn.variable: spn.mu})
        return evidence, spn.value(evidence)
    child_map_results = [max_product(child) for child in spn.children]
    if spn.type == "sum":
        spn = cast(SumNode, spn)
        results = [
            (map_result[0], weight * map_result[1])
            for map_result, weight in zip(child_map_results, spn.weights)
        ]
        return max(results, key=lambda x: x[1])
    # Product
    new_evidence = Evidence()
    value = 1.0
    for result in child_map_results:
        new_evidence.merge(result[0])
        value *= result[1]
    return new_evidence, value


def max_product_with_evidence(
    spn: SPN, evidence: Evidence
) -> Optional[Tuple[Evidence, float]]:
    """Max-product algorithm based on 'Sum-product networks: A new deep architecture
    Inference in Sum-Product Networks'"""
    if spn.type == "leaf":
        spn = cast(Indicator, spn)
        if spn.assignment in evidence[spn.variable]:
            return Evidence({spn.variable: [spn.assignment]}), 1.0
        return None
    if spn.type == "gaussian":
        spn = cast(GaussianNode, spn)
        ev = Evidence({spn.variable: spn.mu})
        return ev, spn.value(ev)
    child_map_results = [
        max_product_with_evidence(child, evidence) for child in spn.children
    ]
    child_map_results = [
        (None, 0.0) if result is None else result for result in child_map_results
    ]
    if spn.type == "sum":
        spn = cast(SumNode, spn)
        results = [
            (map_result[0], weight * map_result[1])
            for map_result, weight in zip(child_map_results, spn.weights)
        ]
        if results:
            return max(results, key=lambda x: x[1])
        return None
    # Product
    new_evidence = Evidence()
    value = 1.0
    for result in child_map_results:
        if result is None:
            return None
        new_evidence.merge(result[0])
        value *= result[1]
    return new_evidence, value


def max_product_with_evidence_and_marginals(
    spn: SPN, evidence: Evidence, marginalized: List[Variable]
) -> Optional[Tuple[Evidence, float]]:
    if spn.type == "leaf":
        spn = cast(Indicator, spn)
        if spn.variable in marginalized:
            return (
                Evidence({spn.variable: list(range(0, spn.variable.n_categories))}),
                1.0,
            )
        if (
            spn.variable in evidence and spn.assignment in evidence[spn.variable]
        ) or spn.variable not in evidence:
            return Evidence({spn.variable: [spn.assignment]}), 1.0
        return None
    # TODO: Check the value for Gaussian Nodes
    child_map_results = [
        max_product_with_evidence_and_marginals(child, evidence, marginalized)
        for child in spn.children
    ]
    child_map_results = [
        (None, 0.0) if result is None else result for result in child_map_results
    ]
    if spn.type == "sum":
        spn = cast(SumNode, spn)
        results = [
            (map_result[0], weight * map_result[1])
            for map_result, weight in zip(child_map_results, spn.weights)
        ]
        if results:
            return max(results, key=lambda x: x[1])
        return None
    # Product
    new_evidence = Evidence()
    value = 1.0
    for result in child_map_results:
        if result is None:
            return None
        new_evidence.merge(result[0])
        value *= result[1]
    return new_evidence, value
