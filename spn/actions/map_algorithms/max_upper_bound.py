from typing import cast, List
from functools import reduce
import operator
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.indicator import Indicator
from spn.actions.map_algorithms.naive import naive, naive_with_evidence_and_marginals
from spn.structs import Variable

def max_upper_bound(spn: SPN, node_limit: int) -> float:
    if spn.nodes() > node_limit:
        if spn.type == 'sum':
            return sum([max_upper_bound(child, node_limit) * weight for child, weight in zip(spn.children, spn.weights)])
        if spn.type == 'product':
            return reduce(operator.mul, [max_upper_bound(child, node_limit) for child in spn.children])
        return 1.0
    return naive(spn)[1]

def max_upper_bound_with_evidence_and_marginals(spn: SPN, node_limit: int, evidence: Evidence, marginals: List[Variable]) -> float:
    if spn.nodes() > node_limit:
        if spn.type == 'sum':
            return sum([max_upper_bound(child, node_limit) * weight for child, weight in zip(spn.children, spn.weights)])
        if spn.type == 'product':
            return reduce(operator.mul, [max_upper_bound(child, node_limit) for child in spn.children])
        return 1.0
    return naive_with_evidence_and_marginals(spn, evidence, marginals)[1]
