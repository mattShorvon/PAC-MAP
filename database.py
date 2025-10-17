"""Database: Definitions for Database class and instance"""

from typing import List, Tuple
from spn.node.base import SPN


class Database:
    """Stores all relevant data of the App. Currently, a dictionary of SPNs and their
    names"""

    def __init__(self):
        self.__data = {}

    @property
    def data(self):
        """The raw dictionary of the database"""
        return self.__data

    def store(self, key, value):
        """Stores the value under the desired key"""
        self.__data[key] = value

    def remove(self, key):
        """Removes data indexed by key"""
        del self.__data[key]

    def get(self, key):
        """Retrieves data indexed by key"""
        return self.__data[key]

    def spns(self) -> List[Tuple[str, SPN]]:
        """Retrieves a List of Tuples with the SPNs and their names"""
        return [
            (spn_name, spn_data_tuple[0])
            for spn_name, spn_data_tuple in self.__data.items()
        ]


DB = Database()
