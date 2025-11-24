"""Classify: Actions for running the spn as a classifier"""

import logging
import numpy
from maua_et_al_experiments.database import DB
from spn.data.partitioned_data import PartitionedData
from spn.node.base import SPN
from spn.actions.base import Action
from spn.utils.evidence import Evidence


class Classify(Action):
    necessary_params = ["spn", "dataset"]
    key = "classify"

    def execute(self):
        """Uses the test partition of data to classify samples"""
        logging.info("Executing Classify action")
        spn = DB.get(self.params["spn"])
        data = self.params["dataset"]
        do_classify(spn, data)


def do_classify(spn: SPN, data: PartitionedData):
    """Use the test data to run classifications with the SPN"""
    variables = data.scope
    last_variable = variables[-1]
    possible_values = range(last_variable.n_categories)
    corrects = 0
    number_of_test_samples, _ = data.test_data.shape
    for test_sample in data.test_data:
        classification_sample = numpy.copy(test_sample)
        probabilities = [0] * len(possible_values)
        for possible_value in possible_values:
            classification_sample[-1] = possible_value
            evidence = Evidence.from_data(classification_sample, variables)
            estimation = spn.log_value(evidence)
            probabilities[possible_value] = estimation
        probabilities = numpy.array(probabilities)
        correct_classification = test_sample[-1]
        spn_classification = numpy.argmax(probabilities)
        if correct_classification == spn_classification:
            corrects += 1
        logging.debug(
            "Should classify as %s, spn classified as %s",
            test_sample[-1],
            numpy.argmax(probabilities),
        )
    print("SPN was correct in %s/%s of cases" % (corrects, number_of_test_samples))
    print("Accuracy: {:.0%}".format(numpy.divide(corrects, number_of_test_samples)))
