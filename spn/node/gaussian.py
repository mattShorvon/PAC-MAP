"""Gaussian: Refers to Gaussian nodes; univariate normal distributions that are
represented as a single leaf on an SPN"""

from typing import Set, Tuple
import numpy
from scipy.stats import norm
from spn.node.base import SPN
from spn.structs import Variable
from spn.utils.evidence import Evidence


class GaussianNode(SPN):
    """A leaf node representing a probability distribution over one variable"""

    def __init__(self, variable: Variable, mu: float, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self._type = "gaussian"
        self.__variable = variable
        self.mu = mu
        self.sigma = sigma

    @property
    def var_id(self) -> int:
        """The identifier of the sole variable this Node represents"""
        return self.__variable.id

    @property
    def variable(self) -> Variable:
        """The variable this Node represents"""
        return self.__variable

    def value(self, evidence: Evidence) -> float:
        if self.sigma == 0:
            if self.mu in evidence[self.__variable]:
                return 1
            return 0

        return sum([
            norm.pdf(x, self.mu, self.sigma)
            for x in evidence[self.__variable]
        ])

    def derivative(self, with_respect_to: Tuple[Variable, int],
            evidence: Evidence) -> float:
        if with_respect_to[0] == self.__variable:
            if evidence.has_var(with_respect_to[0]):
                return self.value(evidence)
        return 0

    def log_value(self, evidence: Evidence):
        return numpy.log(self.value(evidence))

    def scope(self) -> Set[Variable]:
        return {self.__variable}
