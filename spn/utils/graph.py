from typing import cast, Dict
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.indicator import Indicator
from spn.node.base import SPN
from spn.structs import Variable


def binarization(spn: SPN) -> SPN:
    """Returns a new spn with all Sum Nodes 'binarized': Having only two children"""
    if spn.type == "sum":
        spn = cast(SumNode, spn)
        n_children = len(spn.children)
        if n_children > 2:
            if n_children % 2 == 0:
                new_node_1 = SumNode()
                new_node_2 = SumNode()
                half = int(n_children / 2)
                left_children = spn.children[:half]
                left_children_weights = spn.weights[:half]
                right_children = spn.children[half:]
                right_children_weights = spn.weights[half:]
                for child, weight in zip(left_children, left_children_weights):
                    new_node_1.add_child(child, weight / sum(left_children_weights))
                for child, weight in zip(right_children, right_children_weights):
                    new_node_2.add_child(child, weight / sum(right_children_weights))
                spn.children = []
                spn.weights = []
                spn.add_child(binarization(new_node_1), sum(left_children_weights))
                spn.add_child(binarization(new_node_2), sum(right_children_weights))
                return spn

            new_node = SumNode()
            selected_children = spn.children[:-1]
            selected_weights = spn.weights[:-1]
            spn.children = [spn.children[-1]]
            spn.weights = [spn.weights[-1]]
            for child, weight in zip(selected_children, selected_weights):
                new_node.add_child(child, weight / sum(selected_weights))
            spn.add_child(binarization(new_node), 1 - spn.weights[-1])
            return spn

        new_children = [binarization(child) for child in spn.children]
        spn.children = new_children
        return spn
    if spn.type == "product":
        new_children = [binarization(child) for child in spn.children]
        spn.children = new_children
        return spn
    return spn


def full_binarization(spn: SPN) -> SPN:
    """Returns a new spn with all Nodes 'binarized': Having only two children"""
    if spn.type == "sum":
        spn = cast(SumNode, spn)
        n_children = len(spn.children)
        if n_children > 2:
            if n_children % 2 == 0:
                new_node_1 = SumNode()
                new_node_2 = SumNode()
                half = int(n_children / 2)
                left_children = spn.children[:half]
                left_children_weights = spn.weights[:half]
                right_children = spn.children[half:]
                right_children_weights = spn.weights[half:]
                for child, weight in zip(left_children, left_children_weights):
                    new_node_1.add_child(child, weight / sum(left_children_weights))
                for child, weight in zip(right_children, right_children_weights):
                    new_node_2.add_child(child, weight / sum(right_children_weights))
                spn.children = []
                spn.weights = []
                spn.add_child(full_binarization(new_node_1), sum(left_children_weights))
                spn.add_child(
                    full_binarization(new_node_2), sum(right_children_weights)
                )
                assert len(spn.children) <= 2
                return spn

            new_node = SumNode()
            selected_children = spn.children[:-1]
            selected_weights = spn.weights[:-1]
            spn.children = [full_binarization(spn.children[-1])]
            spn.weights = [spn.weights[-1]]
            for child, weight in zip(selected_children, selected_weights):
                new_node.add_child(child, weight / sum(selected_weights))
            spn.add_child(full_binarization(new_node), 1 - spn.weights[-1])
            assert len(spn.children) <= 2
            return spn

        new_children = [full_binarization(child) for child in spn.children]
        spn.children = new_children
        assert len(spn.children) <= 2
        return spn
    if spn.type == "product":
        spn = cast(ProductNode, spn)
        n_children = len(spn.children)
        if n_children > 2:
            if n_children % 2 == 0:
                new_prod_node_1 = ProductNode()
                new_prod_node_2 = ProductNode()
                half = int(n_children / 2)
                for child in spn.children[:half]:
                    new_prod_node_1.add_child(child)
                for child in spn.children[half:]:
                    new_prod_node_2.add_child(child)
                spn.children = []
                spn.add_child(full_binarization(new_prod_node_1))
                spn.add_child(full_binarization(new_prod_node_2))
                assert len(spn.children) <= 2
                return spn

            new_prod_node = ProductNode()
            selected_children = spn.children[:-1]
            other_child = spn.children[-1]
            spn.children = []
            for child in selected_children:
                new_prod_node.add_child(child)
            spn.add_child(full_binarization(other_child))
            spn.add_child(full_binarization(new_prod_node))
            assert len(spn.children) <= 2
            return spn
        new_children = [full_binarization(child) for child in spn.children]
        spn.children = new_children
        assert len(spn.children) <= 2
        return spn
    assert len(spn.children) <= 2
    return spn

def minimization(spn: SPN) -> SPN:
    """Returns the same SPN but with one Indicator ver assignment of variable"""
    indicators = {}
    for var in spn.scope():
        indicators[var] = {}
        for value in range(var.n_categories):
            indicators[var][value] = Indicator(var, value)
    return minimization_r(spn, indicators)


def minimization_r(spn: SPN, indicators: Dict[Variable, Dict[int, Indicator]]) -> SPN:
    """Recursive part of the minimization algorithm"""
    if spn.type == 'leaf':
        return spn
    new_children = []
    for child in spn.children:
        if child.type == 'leaf':
            child = cast(Indicator, child)
            new_children.append(indicators[child.variable][child.assignment])
        else:
            new_children.append(minimization_r(child, indicators))
    spn.children = new_children
    return spn
