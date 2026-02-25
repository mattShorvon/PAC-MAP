"""The graph action exports a representation of an spn in the .graphviz format"""

from typing import Mapping, Any
import logging
from pathlib import Path
from spn.io.file import to_graph_viz
from spn.database import DB
from spn.actions.base import Action

class Graph(Action):
    necessary_params = ['spn', 'filename']
    key = 'graph'

    def execute(self):
        """Prints a representation of the SPN in a .dot graphviz format"""
        logging.info("Executing Graph action")
        spn = DB.get(self.params["spn"])[0]
        to_graph_viz(Path(self.params["filename"]), spn)
