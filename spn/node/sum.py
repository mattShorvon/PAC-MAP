"""Sum: Refers to Sum Nodes: Nodes whose children are associated with weights
and have joint scopes"""

from typing import Tuple, List, Dict
import math
from functools import reduce
import operator
import numpy as np
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.structs import Variable


class SumNode(SPN):
    """A node with children having joint scopes and weights on each edge"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = "sum"
        self.__weights = []
        self.__log_weights = []

    def value(self, evidence: Evidence) -> float:
        return math.fsum(
            [
                child.value(evidence) * weight
                for child, weight in zip(self._children, self.__weights)
            ]
        )

    def eval_r(self, evidence: Evidence, nodes_and_values: Dict[SPN, float]):
        value = 0.0
        for child, weight in zip(self.children, self.__weights):
            if nodes_and_values[child] is None:
                child.eval_r(evidence, nodes_and_values)
            value += nodes_and_values[child] * weight
        nodes_and_values[self] = value

    def derivative(
        self, with_respect_to: Tuple[Variable, int], evidence: Evidence
    ) -> float:
        return math.fsum(
            [
                child.derivative(with_respect_to, evidence) * weight
                for child, weight in zip(self._children, self.__weights)
            ]
        )

    def log_value(self, evidence) -> float:
        values = np.array(
            [
                child.log_value(evidence) + weight
                for child, weight in zip(self._children, self.__log_weights)
            ]
        )
        return np.log(np.sum(np.exp(values)))

    def add_child(self, child: SPN, weight: float):
        """Adds a child to this node with a weighted edge"""
        self._children.append(child)
        self.__weights.append(weight)
        if weight != 0:
            self.__log_weights = np.append(self.__log_weights, np.log(weight))
        else:
            self.__log_weights = np.append(self.__log_weights, -float("Inf"))

    @property
    def weights(self):
        """Returns the list of weights of this Sum Node. The indices of this
        vector are the same for the vector of children, so a zip(node.children,
        node.weights) can produce an iterator for the association of children
        and weights"""
        return self.__weights

    @weights.setter
    def weights(self, new_list_of_weights):
        """Sets the list of weights. There are no checks (right now) because
        it's possible to set the weights before updating the list of children, and
        these checks are left to the Base class"""
        self.__weights = new_list_of_weights
        self.__log_weights = np.log(new_list_of_weights)

    def set_weight_at(self, value: float, index: int, clear_caches=False):
        """Sets the weight of child with the index to the new value
        and update log weight"""
        self.__weights[index] = value
        self.__log_weights[index] = np.log(value)
        if clear_caches:
            self.clear_caches()

    def sample(self, n: int, evidence: Evidence) -> List[Evidence]:
        choices = np.random.choice(self.children, size=n, replace=True, p=self.weights)
        samples = [choice.sample(1, evidence)[0] for choice in choices]
        return samples

    def normalized(self) -> bool:
        return all([child.normalized() for child in self.children]) and math.isclose(
            sum(self.weights), 1.0
        )

    def normalize_weights(self):
        """Change this node's weights to random numbers summing 1"""
        n_weights = len(self.__weights)
        # The dirichlet method returns a matrix 1xN for these parameters
        new_weights_array = np.random.dirichlet(np.ones(n_weights), size=1)[0]
        self.weights = new_weights_array

    def clear_subcaches(self):
        """Recursively clear caches that are defined on subclasses"""
        for child in self.children:
            # Only Sum and Product nodes (for now) have caches
            if child.type in ["sum", "product"]:
                child.clear_subcaches()

    def tensor_value(self, evidences: List[Evidence]) -> np.array:
        values = [0.0] * len(evidences)
        for child, weight in zip(self.children, self.weights):
            child_values = child.tensor_value(evidences) * weight
            for index, child_value in enumerate(child_values):
                values[index] += child_value
        return np.array(values)
