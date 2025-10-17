import itertools
from typing import Tuple, List

from spn.node.base import SPN
from spn.utils.evidence import Evidence


def do_kbt(spn: SPN, k: int) -> Evidence:
    """Function that receives all the K-trees from KBT and returns the best one"""
    best_trees = kbt(spn, k)
    return best_trees[0][0]


def kbt(spn: SPN, k: int) -> List[Tuple[Evidence, float]]:
    """K-Best Tree algorithm
    From: Maximum A Posteriori Inference in Sum-Product Networks (2017)"""
    if spn.type == "sum":
        best_trees = []
        for child, weight in zip(spn.children, spn.weights):
            child_kbts = kbt(child, k)
            for kbt_v in child_kbts:
                best_trees.append((kbt_v[0], weight * kbt_v[1]))
    elif spn.type == "product":
        best_trees = []
        child_kbts = [kbt(child, k) for child in spn.children]
        for tree_to_merge in itertools.product(*child_kbts):
            evidence = Evidence()
            value = 1.0
            for map_result in tree_to_merge:
                evidence.merge(map_result[0])
                value *= map_result[1]
            best_trees.append((evidence, value))

    elif spn.type == "leaf":
        best_trees = [(Evidence({spn.variable: spn.assignment}), 1)]
    best_trees = sorted(best_trees, key=lambda tree: tree[1], reverse=True)
    return best_trees[:k]
