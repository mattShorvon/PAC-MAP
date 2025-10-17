"""Actions related to learning an SPN from data"""

import logging
from pathlib import Path
from spn.learn import gens
from spn.data.partitioned_data import PartitionedData
from database import DB
from spn.actions.base import Action


class Learn(Action):
    necessary_params = ["dataset", "kclusters", "pval", "name"]
    # Since 'gens' is currently the only learning algorithm, naming this action 'SPN'
    # implies this is the only way of obtaining an SPN from data
    key = "SPN"

    def execute(self):
        """Executes the following actions:
            - Loads the dataset
            - Passes the dataset to the learning algorithm
            - Stores the learned SPN with the provided name along with the dataset"""

        logging.info("Executing Learn SPN action")
        data = DB.get(self.params["dataset"])
        spn = gens(
            data.scope, data.data, self.params["kclusters"], self.params["pval"], False
        )
        spn.root = True
        print("SPN learned:", spn.vars(), "vars and", spn.arcs(), "arcs")
        DB.store(self.params["name"], spn)
