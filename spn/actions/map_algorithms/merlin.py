import re
import subprocess
import time
import math
from functools import reduce
import operator
import json
from typing import List, cast, Dict, Tuple, Optional
from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.utils.graph import full_binarization
from spn.structs import Variable
from spn.utils.evidence import Evidence
import os

# To get this path to work (at least on mac), you need to download the merlin 
# repo into Documents as merlin-master, and use its makefile to compile it. 
# Might be ~/Documents/merlin-master/bin instead, whatever works
MERLIN_PATH = os.path.expanduser("~/Documents/merlin-master/bin/merlin")


def merlin(
    spn: SPN,
    evidence_file: str,
    query_file: str,
    uai_file: str,
    ibound: int,
    iterations: int,
    query_vars: List[Variable],
    timeout: int = 600,
) -> Tuple[str, float, Optional[int], Optional[Evidence]]:
    starting_time = time.process_time()
    try:
        print(
            f"{MERLIN_PATH} -a wmb -t MMAP --input-file {uai_file} --evidence-file {evidence_file} --query-file {query_file} --ibound {ibound} --iterations {iterations} --output-format json"
        )
        process_info = subprocess.run(
            [
                MERLIN_PATH,
                "-a",
                "wmb",
                "-t",
                "MMAP",
                "--input-file",
                uai_file,
                "--evidence-file",
                evidence_file,
                "--query-file",
                query_file,
                "--ibound",
                str(ibound),
                "--iterations",
                str(iterations),
                "--output-format",
                "json",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return str(timeout), 0.0, None, None
    except subprocess.CalledProcessError as e:
        print(f"Merlin subprocess failed with return code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    splitted_info = process_info.stdout.strip().split("\n")
    for line in splitted_info:
        if line.startswith("[WMB] + induced width"):
            induced_width = int(line.split(" ")[-1])
    time_line = splitted_info[-5]
    value_line = splitted_info[-4]
    runtime = float(time_line.split(" ")[-2])
    final_value = float(re.sub("[()]", "", value_line.split(" ")[-1]))
    evidence = Evidence()
    json_file = uai_file.split("/")[-1] + ".MMAP.json"
    json_result = json.loads(open(json_file, "r").read())
    var_index = 0
    for the_dict in json_result["solution"]:
        evidence[query_vars[var_index]] = [the_dict["value"]]
        var_index += 1

    return "{:.4f}".format(runtime), final_value, induced_width, evidence


def markov_code(spn: SPN) -> List[str]:
    var_to_index = {}
    node_to_index: Dict[SPN, int] = {}
    parents: Dict[SPN, List[int]] = {}
    returning = []
    variable_index = 0
    for variable in spn.scope():
        var_to_index[variable] = variable_index
        variable_index += 1
    index = variable_index
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
    returning.append(f"{n_chance_nodes + n_decision_nodes}\n")
    for variable in spn.scope():
        returning.append(f"{variable.n_categories} ")
    for _ in range(n_chance_nodes):
        returning.append("2 ")
    returning.append("\n")
    returning.append(f"{n_chance_nodes + n_decision_nodes + 1}\n")
    for variable in spn.scope():
        returning.append(f"1 {var_to_index[variable]}\n")
    for node in reversed(spn.topological_order()):
        line = f"{len(parents[node]) + 1} "
        line += " ".join(reversed([str(x) for x in parents[node]]))
        line += f" {node_to_index[node]}\n"
        returning.append(line)
    returning.append(f"1 {n_chance_nodes + n_decision_nodes - 1}\n")
    #
    for variable in spn.scope():
        returning.append(f"{variable.n_categories} ")
        returning.append(" ".join(["1" for _ in range(variable.n_categories)]))
        returning.append("\n")
    for node in reversed(spn.topological_order()):
        node = cast(Indicator, node)
        if node.type == "leaf":
            returning.append(f"{2 * node.variable.n_categories} ")
            for i in range(node.variable.n_categories):
                # returning.append(f"{i} ")
                if node.assignment == i:
                    returning.append("0 1 ")
                else:
                    returning.append("1 0 ")
            returning.append("\n")
        elif node.type == "product":
            node = cast(ProductNode, node)
            returning.append("8 1 0 1 0 1 0 0 1\n")
        else:  # Sum
            node = cast(SumNode, node)
            returning.append(
                f"8 1 0 {node.weights[1]} {node.weights[0]} {node.weights[0]} {node.weights[1]} 0 1\n"
            )
    returning.append("2 0 1\n")

    return returning


def make_uai_file(spn: SPN, filename: str):
    binarized_spn = full_binarization(spn)
    binarized_spn.fix_scope()
    binarized_spn.fix_topological_order()
    with open(filename, "w") as uai_file:
        uai_file.write("MARKOV\n")
        uai_file.writelines(markov_code(binarized_spn))
