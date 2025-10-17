"""Partition: Partitions datasets into two subdatasets"""

from typing import Tuple
import logging
from database import DB
from spn.actions.base import Action
from spn.data.partitioned_data import PartitionedData
from spn.data.parsed_data import ParsedData


class Partition(Action):
    necessary_params = ["dataset", "proportion", "name1", "name2"]
    key = "partition"

    def execute(self):
        """Uses the parameters to partition a dataset into two"""
        logging.info("Executing Partition action")
        dataset = DB.get(self.params["dataset"])
        data1, data2 = do_partition(dataset, self.params["proportion"])
        DB.store(self.params["name1"], data1)
        DB.store(self.params["name2"], data2)


def do_partition(
    dataset: ParsedData, proportion: float
) -> Tuple[ParsedData, ParsedData]:
    partitioned_dataset = PartitionedData(None, proportion, parsed_data=dataset)
    return (
        ParsedData(
            None, generated=partitioned_dataset.training_data, scope=dataset.scope
        ),
        ParsedData(None, generated=partitioned_dataset.test_data, scope=dataset.scope),
    )
