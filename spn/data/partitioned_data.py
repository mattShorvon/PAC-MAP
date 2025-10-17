"""Partitioned Data Module: Controls data that has been parsed and partitioned
into two sets: One for training, the other supposedly for validating"""
from typing import List
from pathlib import Path
import numpy
from spn.data.parsed_arff import ParsedArff
from spn.data.parsed_data import ParsedData
from spn.structs import Variable


class PartitionedData:
    """PartitionedData: Representation of a data file with part of the data reserved
    as a training sample, and the rest as either validation or test sample"""

    def __init__(
        self,
        dataset_path: Path,
        learning_data_proportion: float,
        parsed_data: ParsedData = None,
    ):
        if parsed_data is None:
            self.__is_numeric = False
            if dataset_path.name.endswith(".arff"):
                parsed_data = ParsedArff(dataset_path)
                self.__is_numeric = parsed_data.is_numeric
            else:
                parsed_data = ParsedData(dataset_path)
        else:
            self.__is_numeric = False

        self.__dataset_path = dataset_path
        self.__learning_data_proportion = learning_data_proportion
        self.__scope = parsed_data.scope
        self.__training_data = parsed_data.data
        self.__test_data = None
        self.__partition_data()

    @property
    def scope(self) -> List[Variable]:
        """The variable scope obtained from the parsed data"""
        return self.__scope

    @property
    def training_data(self):
        """Part of the entire dataset to be used in training"""
        return self.__training_data

    @property
    def test_data(self):
        """Part of the entire dataset to be used in testing"""
        return self.__test_data

    @property
    def is_numeric(self):
        """Returns whether dataset is numeric (or categorical)"""
        return self.__is_numeric

    def __partition_data(self):
        """Splits the training data and the test data by a proportion (a number
        between 0 and 1)"""
        partition = []
        number_of_samples, _ = self.__training_data.shape
        number_of_samples_for_testing = int(
            number_of_samples * (1.0 - self.__learning_data_proportion)
        )
        for _ in range(number_of_samples_for_testing):
            number_of_samples, _ = self.__training_data.shape
            index_of_sample_for_testing = numpy.random.randint(0, number_of_samples - 1)
            partition.append(self.__training_data[index_of_sample_for_testing])
            self.__training_data = numpy.delete(
                self.__training_data, index_of_sample_for_testing, axis=0
            )
        self.__test_data = numpy.array(partition)

    def uniform_sample(self) -> ParsedData:
        """Returns a PartitionedData object with the data being randomly selected (uniform) from the current dataset"""
        data_size = len(self.__training_data)
        indexes_for_new_data = numpy.random.choice(data_size, data_size)
        new_data = self.__training_data[indexes_for_new_data]
        return ParsedData(None, generated=new_data, scope=self.__scope)
