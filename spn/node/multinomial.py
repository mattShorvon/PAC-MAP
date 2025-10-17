"""Multinomial: Refers to Multinomial nodes; univariate distributions that are
represented as a single leaf on an SPN"""

from typing import Set, Tuple
from collections import namedtuple
import numpy
from spn.node.base import SPN
from spn.node.sum import SumNode
from spn.node.indicator import Indicator
from spn.structs import Variable
from spn.utils.evidence import Evidence


Mode = namedtuple("Mode", ["index", "value"])


class MultinomialNode(SPN):
    """A leaf node representing a probability distribution over one variable"""

    def __init__(self, variable: Variable, **kwargs):
        super().__init__(**kwargs)
        self._type = "leaf"
        self.__variable = variable
        counts = kwargs.get("counts")
        if counts is not None:
            distribution = counts + 1
            distribution = distribution / distribution.sum()
        else:
            distribution = kwargs.get("distribution")
        self.__distribution = distribution
        max_index = numpy.argmax(self.__distribution)
        self.__mode = Mode(max_index, self.__distribution[max_index])

    @property
    def var_id(self) -> int:
        """The identifier of the sole variable this Node represents"""
        return self.__variable.id

    @property
    def variable(self) -> Variable:
        """The variable this Node represents"""
        return self.__variable

    @property
    def distribution(self) -> numpy.array:
        """The distribution contains each value this variable can assume and its
        probability"""
        return self.__distribution

    @property
    def mode(self) -> Mode:
        """An easy handler for the highest values of the distribution"""
        return self.__mode

    def value(self, evidence: Evidence) -> float:
        try:
            return sum([self.__distribution[value] for value in evidence[self.__variable]])
        except KeyError:
            raise ValueError(
                'Invoked "value" on a leaf node without providing instantiation \
                of variable'
            )

    def derivative(self, with_respect_to: Tuple[Variable, int], evidence: Evidence) -> float:
        if with_respect_to[0] == self.__variable:
            if evidence.has_var(with_respect_to[0]):
                return self.value(evidence)
        return 0

    def log_value(self, evidence: Evidence):
        return numpy.log(self.value(evidence))

    def scope(self) -> Set[Variable]:
        return {self.__variable}

    def to_sum_node(self) -> SumNode:
        """Returns a Sum Node with a child for each value of the variable
        represented by this MultinomialNode. Each of its values will be represented
        by an Indicator, and the probabilities are the weights of the SumNode."""
        sum_node = SumNode()
        for variable_value, probability in zip(
            range(self.variable.n_categories), self.distribution
        ):
            sum_node.add_child(Indicator(self.variable, variable_value), probability)
        return sum_node
