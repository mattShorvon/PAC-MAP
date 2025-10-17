"""Base: Contains the interface shared by all node types with methods that they
are required to implement"""

import functools
import operator
from itertools import chain
from typing import Dict, List, Tuple, Optional
from queue import Queue
import numpy as np
from spn.structs import Variable
from spn.utils.evidence import Evidence


class SPN:
    """Abstract class containing the minimal functionality for an SPN"""

    def __init__(self, **kwargs):
        self._children = []
        self._parents = []
        self._type = None
        self._eval_cache: Dict[Evidence, List[float]] = {}
        self.__scope = None
        self.__height = None
        self.__root = False
        self.__topological_order: Optional[List["SPN"]] = None
        self.__topological_ids = None
        self.__nodes_cache = None

    @property
    def root(self) -> bool:
        """Returns True if this SPN has been marked as root"""
        return self.__root

    def val_size(self) -> int:
        """Returns the number of complete assignments that this SPN can accept, that is,
        the number of members of the val(X) set, where X is this SPN's scope"""
        return functools.reduce(
            operator.mul, [var.n_categories for var in self.scope()]
        )

    @root.setter
    def root(self, root: bool):
        self.__root = root

    def value(self, evidence: Evidence) -> float:
        """Computes the value of this node given an evidence or partial evidence
        (dict or array)"""
        raise NotImplementedError()

    def eval(self, evidence: Evidence) -> List[float]:
        """Computes the value of each node of the SPN and returns a list
        containing each value in reverse topological order."""
        nodes_and_values = {node: None for node in self.topological_order()}
        self.eval_r(evidence, nodes_and_values)
        return [nodes_and_values[node] for node in reversed(self.topological_order())]

    def eval_r(self, evidence: Evidence, nodes_and_values: Dict["SPN", float]):
        """Recursive helper function for eval"""
        raise NotImplementedError()

    def derivative(
        self, with_respect_to: Tuple[Variable, int], evidence: Evidence
    ) -> float:
        """Returns the derivative of this node at the indicator 'with_respect_to'
        evaluated at the given assignment"""
        raise NotImplementedError()

    def derivatives(
        self, evidence: Evidence, pre_comp_values: Optional[List[float]] = None
    ) -> List[float]:
        """Returns a list containing the derivatives in relation to each node given
        the assignment in reverse topological order"""
        values = pre_comp_values if pre_comp_values is not None else self.eval(evidence)
        topological_order = self.topological_order()
        identifiers = self.__topological_ids
        ders = [0.0] * len(topological_order)
        ders[-1] = 1.0
        for node in topological_order:
            node_derivative = ders[identifiers[node]]
            if node.type == "sum":
                for child, weight in zip(node.children, node.weights):
                    ders[identifiers[child]] += node_derivative * weight
            if node.type == "product":
                children = node.children
                zero_counter = 0
                children_values = []
                for child in children:
                    child_value = values[identifiers[child]]
                    if child_value == 0.0:
                        zero_counter += 1
                    children_values.append(child_value)
                for child in children:
                    other = 0.0
                    if zero_counter == 0:
                        other = values[identifiers[node]] / values[identifiers[child]]
                    elif zero_counter == 1 and values[identifiers[child]] == 0.0:
                        other = np.prod(children_values)
                    ders[identifiers[child]] += node_derivative * other
        return ders

    def derivative_of_assignment(
        self, evidence: Evidence, derivatives: Optional[List[float]] = None
    ) -> Dict[int, List[float]]:
        """Returns the derivatives of the indicators for this assignment"""
        ders = derivatives if derivatives is not None else self.derivatives(evidence)
        reverse_topological_order = reversed(self.topological_order())
        leaf_derivs = {var.id: [0.0] * var.n_categories for var in self.scope()}
        for node_index, node in enumerate(reverse_topological_order):
            if node.type == "leaf":
                try:
                    leaf_derivs[node.var_id][node.assignment] += ders[node_index]
                except AttributeError:  # Constant node
                    for var, values in node.assignments().items():
                        for value in values:
                            leaf_derivs[var.id][value] += ders[node_index]
        return leaf_derivs

    def log_value(self, evidence: Evidence) -> float:
        """Computes the log of the value of this node given an instantiation
        (dict or array) of each value"""
        raise NotImplementedError()

    def clear_scope(self):
        """Sets the scope of this node and all of its children (and so on, recursively) to None"""
        for child in self.children:
            child.clear_scope()
        self.__scope = None

    def fix_scope(self):
        """Computes and caches the scope of this SPN. Any modification on the graph
        will need to call this function again"""
        self.__scope = sorted(
            list(set(chain.from_iterable([child.scope() for child in self._children]))),
            key=lambda x: x.id,
        )

    def scope(self) -> List[Variable]:
        """Returns the scope of this node: The variables that this node points to,
        or the union of the variables from this node's child nodes"""
        if self.__scope is None:
            self.fix_scope()
        return self.__scope

    @property
    def children(self):
        """Returns a list of the child nodes"""
        return self._children

    @children.setter
    def children(self, new_list_of_children: List["SPN"]):
        self._children = new_list_of_children

    @property
    def type(self):
        """The type of the node. Can be either leaf, sum, or product"""
        return self._type

    def fix_topological_order(self):
        """Computes and caches a list of this SPN's nodes in topological order"""
        order = []
        parents = {}
        node_queue = Queue()
        node_queue.put(self)
        parents[self] = 0
        while not node_queue.empty():
            node = node_queue.get()
            for child in node.children:
                if child in parents:
                    parents[child] += 1
                else:
                    parents[child] = 1
                node_queue.put(child)
        node_queue.put(self)
        while not node_queue.empty():
            node = node_queue.get()
            order.append(node)
            for child in node.children:
                parents[child] -= 1
                if parents[child] == 0:
                    node_queue.put(child)
        self.__topological_order = order
        self.__topological_ids = {
            node: number
            for number, node in enumerate(reversed(self.__topological_order))
        }

    def topological_order(self) -> List["SPN"]:
        """Returns a list of nodes in the topological order"""
        if self.__topological_order is None:
            self.fix_topological_order()
        return self.__topological_order

    def height(self) -> int:
        """The height of this node in a tree-like SPN"""
        if self._children:
            return max([child.height() for child in self._children]) + 1
        return 1

    def vars(self) -> int:
        """Returns the number of variables of this SPN"""
        return len(self.scope())

    def arcs(self) -> int:
        """Returns the number of arcs of this SPN"""
        return sum([child.arcs() for child in self._children]) + len(self._children)

    def nodes(self) -> int:
        """Returns the number of nodes of this SPN"""
        if self.__nodes_cache is None:
            self.__nodes_cache = sum([child.nodes() for child in self.children]) + 1
        return self.__nodes_cache

    def possible_evidences(self) -> Evidence:
        """Returns an evidence representing the set of possible evidences acceptable
        by this node"""
        evidence = Evidence()
        for child in self.children:
            evidence.merge(child.possible_evidences())
        return evidence

    def all_marginalized(self) -> Evidence:
        """Returns an evidence with all variables from this SPN marignalized"""
        varset = {var: list(range(var.n_categories)) for var in self.scope()}
        return Evidence(varset)

    def sample(self, n: int, evidence: Evidence) -> List[Evidence]:
        """Returns n random evidences uniformly chosen given the SPN's
        weights and members of the "evidence" parameter set"""
        raise NotImplementedError

    def normalized(self) -> bool:
        """Returns true if the sum of the weights for each Sum Node is 1.0"""
        raise NotImplementedError

    def clear_subcaches(self):
        """Recursively clear caches that are defined on subclasses"""
        raise NotImplementedError

    def clear_caches(self):
        """Clears eval cache, recomputes topological order and scopes"""
        self._eval_cache = {}
        self.fix_topological_order()
        self.clear_scope()
        self.fix_scope()
        self.clear_subcaches()

    def map_problem_size(self) -> int:
        """Returns the size of the MAP problem, which is the product of the number
        of categories for each variable in the scope"""
        return functools.reduce(
            operator.mul, [var.n_categories for var in self.__scope]
        )

    def tensor_value(self, evidences: List[Evidence]) -> np.array:
        """Returns an evaluation of the list of evidences"""
        raise NotImplementedError
