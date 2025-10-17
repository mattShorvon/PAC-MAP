"""Functions for MAP algorithms"""

import logging
from database import DB
from spn.actions.map_algorithms.naive import naive
from spn.actions.map_algorithms.argmax_product import do_argmax_product
from spn.actions.map_algorithms.max_product import do_max_product
from spn.actions.map_algorithms.kbt import do_kbt
from spn.actions.map_algorithms.beam_search import beam_search
from spn.actions.map_algorithms.max_search import (
    max_search,
    forward_checking,
    marginal_checking,
    max_search_with_ordering_and_staging,
)
from spn.actions.map_algorithms.heuristics import ordering
from spn.actions.map_algorithms.branch_and_bound import do_branch_and_bound
from spn.actions.map_algorithms.lagrangian_relaxation import LRDiag
from spn.actions.base import Action


class SPNMap(Action):
    """The MAP action can receive different algorithms, and each algorithm may receive
    different parameters. Currently, this is just a check for each possible algorithm
    and parameter to be sent via the configuration file. When run, it will display the
    found evidence and its value."""

    necessary_params = ["algorithm", "spn"]
    key = "map"

    def execute(self):
        """Finds the answer to the MAP problem for SPNs"""
        header_string = "Executing Map action with algorithm {}".format(
            self.params["algorithm"]
        )
        logging.info(header_string)
        print(header_string)
        spn = DB.get(self.params["spn"])
        if self.params["algorithm"] == "naive":
            evidence = naive(spn)
        elif self.params["algorithm"] == "max-product":
            evidence = do_max_product(spn)
        elif self.params["algorithm"] == "beam-search":
            assert "beam-size" in self.params, "Parameter 'beam-size' not provided"
            beam_size = self.params["beam-size"]
            assert isinstance(beam_size, int), "Parameter 'beam-size' is not an integer"
            assert beam_size > 0, "Parameter 'beam-size' must be greater than zero"
            evidence = beam_search(spn, beam_size)
        elif self.params["algorithm"] == "kbt":
            assert "k" in self.params, "Parameter 'k' not provided"
            param_k = self.params["k"]
            assert isinstance(param_k, int), "Parameter 'k' is not an integer"
            assert param_k > 0, "Parameter 'k' must be greater than zero"
            evidence = do_kbt(spn, param_k)
        elif self.params["algorithm"] == "argmax-product":
            evidence = do_argmax_product(spn)
        elif self.params["algorithm"] == "marginal-checking":
            evidence = max_search(spn, marginal_checking)
        elif self.params["algorithm"] == "forward-checking":
            if "heuristic" in self.params:
                if self.params["heuristic"] == "ordering":
                    evidence = max_search(spn, forward_checking, ordering)
                elif self.params["heuristic"] == "stage":
                    evidence = max_search_with_ordering_and_staging(
                        spn, forward_checking
                    )
                else:
                    raise ValueError(
                        'Unrecognized heuristic for max-search. Only "ordering" and "stage" are acceptable'
                    )
            evidence = max_search(spn, forward_checking)
        elif self.params["algorithm"] == "branch-and-bound":
            evidence = do_branch_and_bound(spn)
        elif self.params["algorithm"] == "lagrangian-relaxation":
            evidence = LRDiag(spn, spn.all_marginalized()).lagrangian_relaxation()
        else:
            error_string = "Algorithm '{}' is not implemented".format(
                self.params["algorithm"]
            )
            logging.error(error_string)
            raise ValueError(error_string)
        complete_report = "  Evidence found: {}\n  Value found: {}".format(
            repr(evidence), spn.value(evidence)
        )
        logging.info(complete_report)
        print(complete_report)
