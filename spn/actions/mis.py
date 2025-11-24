"""MIS: Create an SPN from an instance of the MIS problem"""

import logging
from maua_et_al_experiments.database import DB
from spn.actions.base import Action
from spn.graph.mis import MIS as MISInstance


class MIS(Action):
    necessary_params = ["n", "alpha", "r", "p", "spn"]
    key = "mis"

    def execute(self):
        """Uses the parameters to create an SPN from a MIS intance"""
        logging.info("Executing MIS action")
        mis = MISInstance(
            self.params["n"], self.params["alpha"], self.params["r"], self.params["p"]
        )
        spn = mis.to_spn()
        spn.root = True
        DB.store(self.params["spn"], spn)
