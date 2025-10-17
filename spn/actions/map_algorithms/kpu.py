import re
import subprocess
import time
from typing import List, cast, Dict, Tuple, Optional
from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.utils.graph import full_binarization

SOLVER_PATH = "/home/marcheing/langs/c++/kpu-pp/bin/solve_limid"
TREEWIDTH_LIMIT = 30


def kpu(spn: SPN, timeout: int = 600, full: bool = False) -> Tuple[str, str]:
    temp_filename = "temp.limid"
    binarized_spn = full_binarization(spn)
    binarized_spn.fix_scope()
    binarized_spn.fix_topological_order()
    with open(temp_filename, "w") as limid_file:
        limid_file.write("LIMID\n")
        limid_file.writelines(bayesian_code(binarized_spn))

    n_vars_on_tree = len(spn.scope()) if full else 1
    process_info = subprocess.run(
            [SOLVER_PATH, temp_filename, "0", "2"],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    tree_width_info = re.split(r'\s', process_info.stdout.split('\n')[1])
    tree_width = int(tree_width_info[1])
    weighted_tree_width = int(tree_width_info[-1])

    if tree_width > TREEWIDTH_LIMIT:
        return 'treewidth too large', 'tree_width too large'

    starting_time = time.process_time()
    try:
        process_info = subprocess.run(
            [SOLVER_PATH, temp_filename, str(n_vars_on_tree), "2"],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return str(timeout), str(0.0)
    except subprocess.CalledProcessError:
        return "{:.4f}".format(time.process_time() - starting_time), "error"
    last_line = process_info.stdout.split("\n")[-2]
    last_line_splitted = re.split(" +", last_line)
    runtime = float(last_line_splitted[-2])
    final_value = float(last_line_splitted[-1])

    return "{:.4f}".format(runtime), "{:.4f}".format(final_value)


def bayesian_code(spn: SPN) -> List[str]:
    index = 0
    var_to_index = {}
    node_to_index: Dict[SPN, int] = {}
    parents: Dict[SPN, List[int]] = {}
    returning = []
    variable_index = len(spn.topological_order())
    for variable in spn.scope():
        var_to_index[variable] = variable_index
        variable_index += 1
    for node in reversed(spn.topological_order()):
        if node.type == "leaf":
            node = cast(Indicator, node)
            # The parents of an indicator, on the bayesian network, are the variables
            parents[node] = [var_to_index[node.variable]]
        elif node.type == "sum":
            node = cast(SumNode, node)
            assert len(node.children) <= 2
            parents[node] = [node_to_index[child] for child in node.children]
        else:  # Product
            node = cast(ProductNode, node)
            assert len(node.children) <= 2
            parents[node] = [node_to_index[child] for child in node.children]
        node_to_index[node] = index
        index += 1
    n_chance_nodes = len(spn.topological_order())
    n_decision_nodes = len(spn.scope())
    n_value_nodes = 1
    returning.append(f"{n_chance_nodes} {n_decision_nodes} {n_value_nodes}\n")
    returning.append(re.sub(r" $", "\n", "2 " * n_chance_nodes))
    returning.append(re.sub(r" $", "\n", "2 " * n_decision_nodes))
    for node in reversed(spn.topological_order()):
        line = f"{len(parents[node])} "
        for parent in parents[node]:
            line += f" {parent}"
        line += "\n"
        returning.append(line)
    for variable in spn.scope():
        returning.append("0\n")
    returning.append(f"1 {node_to_index[spn]}\n")
    for node in reversed(spn.topological_order()):
        node = cast(Indicator, node)
        if node.type == "leaf":
            if node.assignment == 0:
                returning.append("4 0 1 1 0\n")
            else:
                returning.append("4 1 0 0 1\n")
        elif node.type == "product":
            node = cast(ProductNode, node)
            returning.append("8 1 0 1 0 1 0 0 1\n")
        else:  # Sum
            node = cast(SumNode, node)
            returning.append(
                f"8 1 0 {1.0 - node.weights[0]} {node.weights[0]} {1.0 - node.weights[1]} {node.weights[1]} 0 1\n"
            )

    returning.append(f"2 0.0 1.0\n")

    return returning
