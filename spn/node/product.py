"""Product: Refers to Productnodes: Nodes whose children have disjoint scopes"""

from typing import Tuple, List, Dict
from functools import reduce
import operator
import numpy as np
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.structs import Variable


class ProductNode(SPN):
    """A node with children having disjoint scopes"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = "product"

    def value(self, evidence: Evidence) -> float:
        return np.prod([child.value(evidence) for child in self._children])

    def eval_r(self, evidence: Evidence, nodes_and_values: Dict[SPN, float]):
        value = 1.0
        for child in self.children:
            if nodes_and_values[child] is None:
                child.eval_r(evidence, nodes_and_values)
            value *= nodes_and_values[child]
        nodes_and_values[self] = value

    def derivative(
        self, with_respect_to: Tuple[Variable, int], evidence: Evidence
    ) -> float:
        return np.prod(
            [child.derivative(with_respect_to, evidence) for child in self._children]
        )

    def log_value(self, evidence: Evidence):
        return sum([child.log_value(evidence) for child in self._children])

    def add_child(self, child: SPN):
        """Adds a child to this node"""
        self._children.append(child)

    def sample(self, n: int, evidence: Evidence) -> List[Evidence]:
        eligible_children = []
        for child in self._children:
            if all([variable in evidence.variables for variable in child.scope()]):
                eligible_children.append(child)
        sample_lists = [child.sample(n, evidence) for child in eligible_children]
        returning_list = []
        for index in range(n):
            ev = Evidence()
            for sample_list in sample_lists:
                ev.merge(sample_list[index])
            returning_list.append(ev)
        return returning_list

    def normalized(self) -> bool:
        return all([child.normalized() for child in self.children])

    def clear_subcaches(self):
        """Recursively clear caches that are defined on subclasses"""
        for child in self.children:
            # Only Sum and Product nodes (for now) have caches
            if child.type in ["sum", "product"]:
                child.clear_subcaches()

    def tensor_value(self, evidences: List[Evidence]) -> np.array:
        values = [1.0] * len(evidences)
        for child in self.children:
            child_values = child.tensor_value(evidences)
            for index, value in enumerate(child_values):
                values[index] *= value
        return np.array(values, dtype=float)
