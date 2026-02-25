"""Copy: Action for copying an entry from the database"""

import logging
import copy
from spn.database import DB
from spn.actions.base import Action

class Copy(Action):
    """Copy: Action for copying database entries"""
    necessary_params = ['source', 'destination']
    key = 'copy'

    def execute(self):
        """Copies whatever data is indexed by 'source' in the database to
        'destination'"""
        logging.info("Executing Copy action")
        data = DB.get(self.params['source'])
        if data is None:
            logging.error(
                "No data indexed in key %s", self.params['source']
            )
            return
        DB.store(self.params['destination'], copy.deepcopy(data))
