"""Report: Obtain readable data about the algorithms and SPNs"""

from typing import Any, Tuple, Callable
import time
import logging
from database import DB
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.map_algorithms.naive import naive
from spn.actions.map_algorithms.argmax_product import do_argmax_product
from spn.actions.map_algorithms.max_product import do_max_product
from spn.actions.map_algorithms.kbt import do_kbt
from spn.actions.map_algorithms.beam_search import beam_search
from spn.actions.map_algorithms.max_search import max_search, forward_checking, marginal_checking
from spn.actions.map_algorithms.branch_and_bound import do_branch_and_bound
from spn.actions.base import Action


class Report(Action):
    necessary_params = ["beam-size", "k", "spn", "repeats"]
    key = "report"

    def execute(self):
        """Runs all the map algorithms repeatedly and returns their results as
        Evidences"""
        logging.info("Running MAP report on spn: %s", self.params["spn"])
        beam_sz = self.params["beam-size"]
        k = self.params["k"]
        spn = DB.get(self.params["spn"])[0]
        repeats = self.params["repeats"]
        algorithm_params = [
            ("Naive Map", naive, None),
            ("Max Product / Best Tree", do_max_product, None),
            ("Argmax Product", do_argmax_product, None),
            ("Beam Search with Beam Size = {}".format(beam_sz), beam_search, beam_sz),
            ("K-Best Tree with K = {}".format(k), do_kbt, k),
            ("Max Search with Marginal Checking", max_search, marginal_checking),
            ("Max Search with Forward Checking", max_search, forward_checking),
            ("Branch and Bound", do_branch_and_bound, None),
        ]
        for alg_name, alg, param in algorithm_params:
            result, time_elapsed = run_algorithm_and_check_time(
                spn, alg, param, repeats
            )
            print("Algorithm: {}".format(alg_name))
            print("  Evidence Found: {}".format(repr(result)))
            print("  Value Found: {}".format(spn.value(result)))
            print("  Time elapsed: {}s".format(time_elapsed))


class AlgorithmReport(Action):
    """Tests an algorithm on all stored SPNs and saves results on a .csv file"""
    necessary_params = ["repeats", "algorithm", "parameters", "filename"]
    key = "algorithm-report"

    def execute(self):
        """Run the map algorithm repeatedly with the parameters and saves a .csv file"""
        algorithm_name = self.params["algorithm"]
        logging.info("Running Algorithm report for algorithm %s", algorithm_name)
        beam_sz = None
        k = None
        if algorithm_name == "beam-search":
            if not "beam-size" in self.params["parameters"]:
                logging.error("Missing beam_size parameter for algorithm report")
                print("Missing beam-size parameter for beam-search")
            beam_sz = self.params["parameters"]["beam-size"]
        if algorithm_name == "kbt":
            if not "k" in self.params["parameters"]:
                logging.error("Missing k parameter for algorithm report")
                print("Missing k parameter for kbt")
            k = self.params["parameters"]["k"]
        repeats = self.params["repeats"]
        algorithm_params = {
            "naive": (naive, None),
            "max-product": (do_max_product, None),
            "argmax-product": (do_argmax_product, None),
            "beam-search": (beam_search, beam_sz),
            "kbt": (do_kbt, k),
            "marginal-checking": (max_search, marginal_checking),
            "forward-checking": (max_search, forward_checking),
            "branch-and-bound": (do_branch_and_bound, None),
        }

        algorithm, param = algorithm_params[algorithm_name]

        with open(self.params["filename"], 'w') as file:
            file.write(f'spn,nodes,edges,height,variables,time\n')
            for name, spn in DB.spns():
                print(f"Report for algorithm {algorithm_name} on SPN {name}")
                _, time_elapsed = run_algorithm_and_check_time(
                    spn, algorithm, param, repeats
                )
                file.write(f'{name},{spn.nodes()},{spn.arcs()},{spn.vars()},{time_elapsed}\n')


def run_algorithm_and_check_time(
    spn: SPN, algorithm: Callable[[SPN, Any], Evidence], param: Any = None, repeats=1
) -> Tuple[Evidence, int]:
    """Run the algorithm parameter 'repeats' times with the spn and the given param
    (if not None), returning the Evidence found and the average elapsed time"""
    diff_time = 0
    for _ in range(repeats):
        starting_time = time.process_time()
        result = algorithm(spn) if param is None else algorithm(spn, param)
        diff_time += time.process_time() - starting_time
    return result, diff_time / repeats
