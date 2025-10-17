"""Categorical Test: Contains independence test for two categorical random
variables using gtest."""
import numpy
from spn.indep.gtest import gtest, gtest_unoptim


def categorical_test(sample, sample2, pval):
    """Test if two variable samples are independent using gtest."""
    sample_n_categories = sample.variable.n_categories
    sample2_n_categories = sample2.variable.n_categories

    # The count_matrix will count the amount of samples that have a
    # determined value. For example, if you have the samples:
    # 0 1
    # 0 0
    # 0 1
    # 1 0
    # 1 2
    # The first var has 2 categories and the second one has 3.
    # The matrix will end up as:
    # | 1 2 0 |
    # | 1 0 1 |
    count_matrix = numpy.zeros(
        [sample_n_categories, sample2_n_categories], int
    )

    for k in range(len(sample.data)):
        count_matrix[sample.data[k]][sample2.data[k]] += 1

    try:
        indep = gtest(count_matrix, pval)
    except ValueError:
        # Table of expectancies has a zero, which means the data is
        # small enough that unoptimized calculations can be used
        indep = gtest_unoptim(
            sample_n_categories, sample2_n_categories, count_matrix, pval
        )

    return indep
