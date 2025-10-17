"""Parsed Arff Module -- Contains the ParsedArff class for parsing arff files"""

from typing import List
import arff
import numpy
from spn.structs import Variable


class ParsedArff:
    """ParsedArff: Representation of an arff file in numpy arrays"""

    def __init__(self, dataset_path):
        self.__dataset_path = None
        self.__scope = None
        self.__data = None
        self.__is_numeric = False
        self.change_dataset(dataset_path)

    @property
    def dataset_path(self):
        """Returns Path value representing the path to the current dataset file"""
        return self.__dataset_path

    @property
    def is_numeric(self):
        """Returns whether dataset is numeric (or categorical)"""
        return self.__is_numeric

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
        with open(dataset_path) as dataset_file:
            data = arff.load(dataset_file)

            has_categorical = False
            has_numeric = False

            self.__scope = []
            class_to_int = {}
            for i, attr in enumerate(data['attributes']):
                n_categories = 0

                if isinstance(attr[1], list):
                    # Variable is categorical.
                    has_categorical = True
                    n_categories = len(attr[1])
                    class_to_int[i] = {}
                    for j, value in enumerate(attr[1]):
                        class_to_int[i][value] = j
                else:
                    has_numeric = True

                self.__scope.append(Variable(i, n_categories))

            if has_categorical and has_numeric:
                raise ValueError(
                    "Dataset contains mixed (categorical and numeric) attributes."
                )

            if has_numeric:
                self.__is_numeric = True

            self.__data = []
            for sample in data['data']:
                for j, value in enumerate(sample):
                    if j in class_to_int:
                        # Convert category to integer ID.
                        sample[j] = class_to_int[j][value]

                self.__data.append(sample)

            self.__data = numpy.array(self.__data)
            self.__dataset_path = dataset_path
