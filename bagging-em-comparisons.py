import gc
import os
from os import listdir
from os.path import isfile, join
import time
import sys
from functools import reduce
import operator
from pathlib import Path
from spn.learn import gens
from spn.actions.map_algorithms.branch_and_bound import do_branch_and_bound
from spn.data.partitioned_data import ParsedData
from spn.actions.map_algorithms.max_search import (
    max_search_with_ordering,
    forward_checking,
)
from spn.actions.partition import do_partition
from spn.actions.em import em_with_restarts
from spn.actions.bag import do_bagging_with_em
from spn.node.base import SPN
from spn.actions.likelihood import ll_from_data

REPEATS = 30
EM_RESTARTS = 10

KCLUSTERS = 3
PVAL = 0.6
TRAINING_DATA_PROPORTION = 0.7

LAPLACIAN = 1e-3
ITERATIONS = 10
EPSILON = 1e-2


def problem_size(spn: SPN) -> int:
    return reduce(operator.mul, [var.n_categories for var in spn.scope()])


def measure(spn: SPN, algorithm, algorithm_name: str, parameter=None) -> float:
    total_time = 0.0
    for _ in range(REPEATS):
        gc.collect()
        gc.disable()
        starting_time = time.process_time()
        result = algorithm(spn) if parameter is None else algorithm(spn, parameter)
        total_time += time.process_time() - starting_time
        gc.enable()
    mean_time = total_time / float(REPEATS)
    # print(f"      Result for {algorithm_name}:", result, " => ", spn.value(result))
    # print(f"      Time for {algorithm_name}:", mean_time)
    # sys.stdout.flush()
    return mean_time


def do_experiments(
    spn_name: str, spn: SPN, training_data: ParsedData, test_data: ParsedData
):
    training_data_evidence = training_data.generate_evidences()
    test_data_evidence = test_data.generate_evidences()
    # print(f'  Experiments for spn {spn_name}')
    ll_train = "{:.4f}".format(ll_from_data(spn, training_data_evidence))
    ll_test = "{:.4f}".format(ll_from_data(spn, test_data_evidence))
    # print(f"    Height {spn.height()}")
    # print(f"    Vars {len(spn.scope())}")
    # print(f"    Problem Size: {problem_size(spn)}")
    # print(f"    Nodes {spn.nodes()}")
    # print(f"      LL Train: {}")
    # print(f"      LL Test: {}")
    time_bb = "{:.4f}".format(measure(spn, do_branch_and_bound, "B&B"))
    time_max = "{:.4f}".format(
        measure(spn, max_search_with_ordering, "MAX", parameter=forward_checking)
    )
    print(
        f"{spn_name} & {len(spn.scope())} & {problem_size(spn)} & {spn.nodes()} & ${ll_train}$ & ${ll_test}$ & ${time_bb}$ & ${time_max}$ \\\\ \hline"
    )
    sys.stdout.flush()


DATA_FOLDER = "data"

data_filenames = [f for f in listdir(DATA_FOLDER) if isfile(join(DATA_FOLDER, f))]
pairs = []
for filename in data_filenames:
    location = join(DATA_FOLDER, filename)
    size = os.path.getsize(location)
    pairs.append((size, filename))
pairs.sort(key=lambda s: s[0])
data_filenames = [pair[1] for pair in pairs]
datas = []
for data_filename in data_filenames:
    full_data = ParsedData(Path(join(DATA_FOLDER, data_filename)))
    datas.append((data_filename, do_partition(full_data, TRAINING_DATA_PROPORTION)))


print(
    f"Name & Scope & Problem Size & Nodes & LL train & LL test && Time BB & Time MAX \\\\ \hline"
)
for name, (training_data, test_data) in datas:
    scope = training_data.scope
    spn = gens(scope, training_data.data, KCLUSTERS, PVAL, False)
    name = name.replace(".data", "")
    bagged_spns = {
        n_bags: do_bagging_with_em(
            training_data,
            test_data,
            KCLUSTERS,
            PVAL,
            n_bags,
            ITERATIONS,
            LAPLACIAN,
            EPSILON,
            EM_RESTARTS,
        )
        for n_bags in [10, 20, 30, 40]
    }
    spn_em = em_with_restarts(
        spn, training_data, test_data, ITERATIONS, LAPLACIAN, EPSILON, EM_RESTARTS
    )
    do_experiments(f"{name}", spn, training_data, test_data)
    for n_bags, bagged_spn in bagged_spns.items():
        do_experiments(
            f"Bag {n_bags} {name}", bagged_spns[10], training_data, test_data
        )
    do_experiments(f"{name} EM", spn_em, training_data, test_data)
