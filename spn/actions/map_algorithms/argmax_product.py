from typing import Tuple, cast, List
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.node.indicator import Indicator
from spn.structs import Variable


def do_argmax_product(spn: SPN) -> Evidence:
    """Caller for argmax_product to adequate returned value to be only
    the Evidence"""
    # TODO: Change the recursive argmax_product function to return only the
    # evidence
    return argmax_product(spn)[0]


def argmax_product(spn: SPN) -> Tuple[Evidence, float]:
    """Argmax-product algorithm based on 'Approximation Complexity of Maximum A
    Posteriori Inference in Sum-Product Networks'"""
    if spn.type == "sum":
        results = []
        for child in spn.children:
            evidence = argmax_product(child)[0]
            results.append((evidence, spn.value(evidence)))
        return max(results, key=lambda x: x[1])
    if spn.type == "product":
        evidence = Evidence()
        child_evidences = [argmax_product(child)[0] for child in spn.children]
        for child_evidence in child_evidences:
            evidence.merge(child_evidence)
        return (evidence, spn.value(evidence))
    if spn.type == "gaussian":
        evidence = Evidence({spn.variable: spn.mu})
        return evidence, spn.value(evidence)
    # Leaf (indicator)
    spn = cast(Indicator, spn)
    return Evidence({spn.variable: [spn.assignment]}), 1


def argmax_product_with_evidence_and_marginalized(
    spn: SPN, evidence: Evidence, marginalized: List[Variable]
) -> Tuple[Evidence, float]:
    """Argmax-product algorithm based on 'Approximation Complexity of Maximum A
    Posteriori Inference in Sum-Product Networks'"""
    if spn.type == "sum":
        results = []
        for child in spn.children:
            result_evidence, result_value = argmax_product_with_evidence_and_marginalized(
                child, evidence, marginalized
            )
            if result_value > 0.0:
                to_store_value = spn.value(result_evidence)
            else:
                to_store_value = 0.0
            results.append((result_evidence, to_store_value))
        return max(results, key=lambda x: x[1])
    if spn.type == "product":
        new_evidence = Evidence()
        value = 1.0
        child_results = [
            argmax_product_with_evidence_and_marginalized(child, evidence, marginalized)
            for child in spn.children
        ]
        for child_evidence, child_value in child_results:
            new_evidence.merge(child_evidence)
            value *= child_value
        return (new_evidence, value)
    # Leaf (indicator)
    spn = cast(Indicator, spn)
    if spn.variable in marginalized:
        return Evidence({spn.variable: list(range(spn.variable.n_categories))}), 1.0
    if (spn.variable in evidence and spn.assignment in evidence[spn.variable]) or ( 
        spn.variable not in evidence
    ):
        return Evidence({spn.variable: [spn.assignment]}), 1.0
    return Evidence(), 0.0
