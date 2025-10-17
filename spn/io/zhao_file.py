"""SPN Zhao File: Methods for writing and reading SPNs from and to the Zhao's SPN file type:
https://github.com/KeiraZhao/SPN"""

from pathlib import Path
import re
from queue import Queue
from collections import namedtuple
from spn.node.base import SPN
from spn.node.gaussian import GaussianNode
from spn.node.indicator import Indicator
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.structs import Variable


def to_zhao_file(spn: SPN, file_path: Path):
    """Saves the SPN into a Zhao SPN file"""
    topological_order = spn.topological_order()
    node_to_id = {
        node: i for i, node in zip(range(len(topological_order)), topological_order)
    }
    with open(file_path, "w") as the_file:
        write_node(spn, the_file, node_to_id)
    print(f"Saved file {file_path}")


def from_zhao_file(file_path: Path) -> SPN:
    """Saves the SPN into a Zhao SPN file"""
    ids_to_nodes = {}
    with open(file_path, "r") as the_file:
        for line in the_file.readlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            splitted_line = line.split(",")
            node_id1 = int(splitted_line[0])
            if splitted_line[1] == "SUM":
                ids_to_nodes[node_id1] = SumNode()
            elif splitted_line[1] == "PRD":
                ids_to_nodes[node_id1] = ProductNode()
            elif splitted_line[1] == "LEAVE":
                variable = Variable(int(splitted_line[2]), 2)
                node = SumNode()
                node.add_child(Indicator(variable, 0), float(splitted_line[3]))
                node.add_child(Indicator(variable, 1), float(splitted_line[4]))
                ids_to_nodes[node_id1] = node
            else:
                node1 = ids_to_nodes[node_id1]
                node2 = ids_to_nodes[int(splitted_line[1])]
                if len(splitted_line) > 2:
                    weight = float(splitted_line[2])
                    node1.add_child(node2, weight)
                else:
                    node1.add_child(node2)
    return ids_to_nodes[0]


def write_node(node: SPN, the_file, node_to_id, parent_id=None, weight=None):
    if isinstance(node, SumNode):
        if all([isinstance(child, Indicator) for child in node.children]):
            weight_false, weight_true = (
                (node.weights[0], node.weights[1])
                if node.children[0].assignment == 0
                else (node.weights[1], node.weights[0])
            )
            the_file.write(
                f"{node_to_id[node]},BINNODE,{node.scope()[0].id},{weight_false},{weight_true}\n"
            )
        else:
            the_file.write(f"{node_to_id[node]},SUM\n")
            for child, child_weight in zip(node.children, node.weights):
                write_node(child, the_file, node_to_id, node_to_id[node], child_weight)
    elif isinstance(node, ProductNode):
        the_file.write(f"{node_to_id[node]},PRD\n")
        for child in node.children:
            write_node(child, the_file, node_to_id, node_to_id[node])
    if parent_id is not None:
        if weight is not None:
            the_file.write(f"{parent_id},{node_to_id[node]},{weight}\n")
        else:
            the_file.write(f"{parent_id},{node_to_id[node]}\n")
