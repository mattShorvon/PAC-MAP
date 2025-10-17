"""A constant node can be created by the "staging" heuristic in certain MAP
algorithms. This node is an indicator that only returns one value, no
matter the evidence"""

from typing import List, Dict
import numpy
from spn.node.base import SPN
from spn.structs import Variable
from spn.utils.evidence import Evidence


class Constant(SPN):
    """A leaf node representing a number."""

    def __init__(self, value: float, scope: List[Variable], assignments: Evidence, **kwargs):
        super().__init__(**kwargs)
        self._type = "leaf"
        self.__value = value
        self.__scope = scope
        self.__assignments = assignments

    def eval_r(self, _: Evidence, nodes_and_values: Dict[SPN, float]):
        nodes_and_values[self] = self.__value

    def scope(self) -> List[Variable]:
        return self.__scope

    def value(self, _: Evidence) -> float:
        return self.__value

    def assignments(self) -> Evidence:
        return self.__assignments

    def log_value(self, evidence: Evidence) -> float:
        return numpy.log(self.value(evidence))
