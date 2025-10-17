"""Module App: Controls the main application"""
import json
import logging
import time
from pathlib import Path
import numpy
from spn.actions.index import ACTIONS


class App:
    """App: Class that handles configurations and calling of other functionalities"""

    CONFIG_FILE_PATH = Path("config/spn.json")

    def __init__(self):
        self.__version = "0.0.0-alpha"
        self.__config = None

    def start(self):
        """Runs the app"""
        self.__config = json.loads(open(self.CONFIG_FILE_PATH).read())
        self.__config_logging()
        self.__config_numpy()
        self.__execute_actions()

    def __config_logging(self):
        """Configure the logging file according to the
        information obtained from the config script"""
        logging.basicConfig(
            filename=self.__config["log-path"],
            filemode="w",
            format="%(asctime)s %(levelname)s:%(message)s",
            level=logging.DEBUG,
        )

    def __config_numpy(self):
        """Configure the numpy random seed"""
        numpy.random.seed(self.__config["random"]["seed"])

    def __execute_actions(self):
        """Calls the respective method for executing every action loaded from the
        config"""
        actions = self.__config["actions"]
        for action_dict in actions:
            action_tuple = action_dict.popitem()
            action_name = action_tuple[0]
            action_params = action_tuple[1]
            starting_time = time.process_time()
            ACTIONS[action_name](action_params)
            logging.info(
                "Action %s took %s seconds",
                action_name,
                time.process_time() - starting_time,
            )

    @property
    def version(self):
        """Version should change on each major patch"""
        return self.__version
