"""SPN file: Methods for writing and reading SPNs from and to the spn file type"""

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

BFSPair = namedtuple("BFSPair", ["spn", "pname", "weight"])


def to_file(spn: SPN, file_path: Path):
    """Saves the SPN in a file with a proper formatting to be read in the future"""
    topological_order = spn.topological_order()
    current_id = 0
    node_id_dict = {}
    for node in topological_order:
        node_id_dict[node] = current_id
        current_id += 1
    with open(file_path, "w") as spn_file:
        for node in reversed(topological_order):
            if node.type == "leaf":
                spn_file.write(
                    "indicator %s %s %s"
                    % (node_id_dict[node], node.var_id, node.assignment)
                )
            elif node.type == "gaussian":
                spn_file.write(
                    "gaussian %s %s %f %f"
                    % (node_id_dict[node], node.var_id, node.mu, node.sigma)
                )
            elif node.type == "sum":
                spn_file.write("+ %s" % (node_id_dict[node]))
                for child, weight in zip(node.children, node.weights):
                    spn_file.write(" %s %s" % (node_id_dict[child], weight))
            else:
                spn_file.write("* %s" % (node_id_dict[node]))
                for child in node.children:
                    spn_file.write(" %s" % node_id_dict[child])
            spn_file.write("\n")


def from_file(file_path: Path) -> SPN:
    """Creates an SPN from an exported SPN file"""
    last = None
    with open(file_path, "r") as spn_file:
        id_node_dict = {}
        var_id_dict = {}  # Holds variables and their ids
        for line in spn_file.readlines():
            line_elems = re.split(r"\s", line)[:-1]
            node_type = line_elems[0]
            the_id = int(line_elems[1])
            if node_type in ("indicator", "l"):
                var_id = int(line_elems[2])
                assignment = int(line_elems[3])
                if var_id in var_id_dict:
                    # The number of categories for each variable is obtained through
                    # the number of assignments
                    # A Variable with a category without assignments has unknown
                    # probability values, so it's not generated
                    if assignment + 1 > var_id_dict[var_id].n_categories:
                        var_id_dict[var_id] = Variable(var_id, assignment + 1)
                else:
                    var_id_dict[var_id] = Variable(var_id, assignment + 1)
                node = Indicator(var_id_dict[var_id], assignment)
            elif node_type == "gaussian":
                var_id = int(line_elems[2])
                mu = float(line_elems[3])
                sigma = float(line_elems[4])
                if var_id not in var_id_dict:
                    var_id_dict[var_id] = Variable(var_id, 1)
                node = GaussianNode(var_id_dict[var_id], mu, sigma)
            elif node_type == "+":
                node = SumNode()
                iterator = iter(line_elems[2:])
                for node_id in iterator:
                    weight = float(next(iterator))
                    node.add_child(id_node_dict[int(node_id)], weight)
            else:
                node = ProductNode()
                for node_id in line_elems[2:]:
                    node.add_child(id_node_dict[int(node_id)])
            id_node_dict[the_id] = node
            last = node
    for _, node in id_node_dict.items():
        if node.type == "leaf":
            node.variable = var_id_dict[node.variable.id]
    # The last one is the root, so it's returned
    last.root = True
    return last


def to_graph_viz(file_path: Path, spn: SPN):
    """Writes to file_path a representation in graph_viz of this SPN"""
    with open(file_path, "w") as graph_file:
        graph_file.write("graph {\n")
        if spn.type == "leaf":
            graph_file.write("X1 [label=<X<sub>1</sub>,shape=circle>];\n}")
            return
        nvars, nsums, nprods = 0, 0, 0
        node_queue = Queue()
        node_queue.put(BFSPair(spn, "", -1.0))
        while not node_queue.empty():
            current = node_queue.get()
            pname = current.pname
            weight = current.weight
            name = "N"
            current_type = current.spn.type
            weights = []

            if current_type == "sum":
                name = "S%d" % nsums
                graph_file.write('%s [label="+",shape=circle];\n' % name)
                nsums += 1
            elif current_type == "product":
                name = "P%d" % nprods
                graph_file.write("%s [label=<&times;>,shape=circle];\n" % name)
                nprods += 1

            # If pname is empty, then it is the root node. Else link parent node
            # to current node
            if pname:
                if weight >= 0.0:
                    graph_file.write('%s -- %s [label="%s"];\n' % (pname, name, weight))
                else:
                    graph_file.write("%s -- %s\n" % (pname, name))

            if current_type == "sum":
                weights = current.spn.weights

            # For each children, run the BFS
            for child_index, child in enumerate(current.spn.children):
                if child.type == "leaf":
                    child_name = "X%s" % nvars
                    child_var = next(iter(child.scope()))
                    graph_file.write(
                        "%s [label=<X<sub>%s</sub>=%s>,shape=circle];\n"
                        % (child_name, child_var.id, child.assignment)
                    )
                    nvars += 1

                    if current_type == "sum":
                        graph_file.write(
                            '%s -- %s [label="%s"]\n'
                            % (name, child_name, weights[child_index])
                        )
                    else:
                        graph_file.write("%s -- %s\n" % (name, child_name))
                else:
                    child_weight = -1.0
                    if weights:
                        child_weight = weights[child_index]
                    node_queue.put(BFSPair(child, name, child_weight))
        graph_file.write("}")
