"""Parsed Data Module -- Contains the ParsedData class for parsing data files"""
import logging

import re
from pathlib import Path
from typing import Union, List
import numpy
from spn.structs import Variable
from spn.utils.evidence import Evidence


def string_to_data_point(string: str) -> Union[int, float]:
    """A data póint can be either an int or a float. This will attempt to
    convert the data to int first, and then will try float if it doesn't work"""
    try:
        return int(string)
    except ValueError:
        return float(string)


class ParsedData:
    """ParsetData: Representation of a data file in numpy arrays"""

    def __init__(self, dataset_path, generated=None, scope=None):
        if generated is None:
            self.__dataset_path = None
            self.__scope = None
            self.__data = None
            self.change_dataset(dataset_path)
        else:
            self.__data = generated
            self.__scope = scope
        self.__data_as_evidences = None

    @property
    def dataset_path(self):
        """Returns Path value representing the path to the current dataset file"""
        return self.__dataset_path

    @property
    def data(self):
        """A numpy array with all data from the file, ordered by variable. This needs
        the variable ids to be ordered similar to the ordering in an array"""
        return self.__data

    @property
    def scope(self) -> List[Variable]:
        """A list with the variables on this data"""
        return self.__scope

    def change_dataset(self, dataset_path):
        """Remove previous dataset data and assigns new data from new dataset"""
        try:
            with open(dataset_path) as dataset_file:
                self.__scope = []
                self.__data = []
                for line in dataset_file.readlines():
                    line_data = re.split(",| ", line)
                    if line_data[0] == "var":
                        # Variable declaration: "var id categories"
                        var_id = int(line_data[1])
                        number_of_categories = int(line_data[2])
                        self.__scope.append(Variable(var_id, number_of_categories))
                    else:
                        # Data sample line. Only numbers; one for each variable
                        data_sample = list(map(string_to_data_point, line_data))
                        self.__data.append(data_sample)
            self.__data = numpy.array(self.__data)
            self.__dataset_path = dataset_path
        except OSError as err:
            logging.error(err)
            raise err

    def uniform_sample(self) -> "ParsedData":
        """Returns a ParsedDatga object with the data being randomly selected (uniform)
        from the current dataset"""
        data_size = len(self.__data)
        indexes_for_new_data = numpy.random.choice(data_size, data_size)
        new_data = self.__data[indexes_for_new_data]
        return ParsedData(None, generated=new_data, scope=self.__scope)

    def generate_evidences(self) -> List[Evidence]:
        if self.__data_as_evidences is None:
            self.__data_as_evidences = [
                Evidence.from_data(data_instance, self.__scope)
                for data_instance in self.__data
            ]
        return self.__data_as_evidences

    def to_file(self, filename: Path, pyspn_format=False):
        with open(filename, "w") as the_file:
            for variable in self.__scope:
                if pyspn_format:
                    the_file.write(f"var {variable.id} {variable.n_categories}\n")
            for data_instance in self.__data:
                the_file.write(",".join([str(x) for x in data_instance]))
                the_file.write("\n")
        print(f"Wrote file {filename}")
