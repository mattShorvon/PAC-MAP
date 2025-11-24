"""Indicator refers to Indicator nodes; Activated if the variable has a
determined value"""

from typing import List, Dict
import numpy as np
from spn.node.base import SPN
from spn.structs import Variable
from spn.utils.evidence import Evidence


class Indicator(SPN):
    """A leaf node representing the assignment of a single variable."""

    def __init__(self, variable: Variable, assignment: int, **kwargs):
        super().__init__(**kwargs)
        self._type = "leaf"
        self.__variable = variable
        self.__assignment = assignment

    @property
    def var_id(self) -> int:
        """The identifier of the sole variable this Node represents"""
        return self.__variable.id

    @property
    def variable(self) -> Variable:
        """The variable this Node represents"""
        return self.__variable

    @variable.setter
    def variable(self, new_var: Variable):
        """Sets the variable field to a new Variable"""
        self.__variable = new_var

    @property
    def assignment(self) -> int:
        """The assignment contains the value of the variable represented by this node"""
        return self.__assignment

    def scope(self) -> List[Variable]:
        return [self.__variable]

    def value(self, evidence: Evidence) -> float:
        # return 1.0 if self.__assignment in evidence[self.__variable] else 0.0 # causes typeerror: argument of type 'int' is not iterable
        if self.variable not in evidence: 
            return 1.0
        return 1.0 if self.assignment in evidence[self.variable] else 0.0

    def eval_r(self, evidence: Evidence, nodes_and_values: Dict[SPN, float]):
        nodes_and_values[self] = self.value(evidence)

    def log_value(self, evidence: Evidence) -> float:
        value = self.value(evidence)
        if value == 0:
            return -np.inf
        return np.log(value)

    def possible_evidences(self) -> Evidence:
        return Evidence({self.__variable: [self.__assignment]})

    def sample(self, n: int, evidence: Evidence) -> List[Evidence]:
        if self.assignment in evidence[self.variable]:
            return [Evidence({self.variable: [self.assignment]})] * n
        raise ValueError(
            "Called sample with evidence that's not part of this node's scope"
        )

    def normalized(self) -> bool:
        return True

    def tensor_value(self, evidences: List[Evidence]) -> np.array:
        return np.array([self.value(evidence) for evidence in evidences])
