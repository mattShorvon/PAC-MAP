"""Gtest: The maximum likelihood statistical significance test. Contains only
this function"""
import numpy
from scipy.stats import chi2, chi2_contingency, contingency


def gtest(contingency_table: numpy.array, sigval: float) -> bool:
    """Computes the G-test for independence between two variables using a dataset"""
    _, comparison, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
    return comparison >= sigval


def gtest_unoptim(
    var1_n_categories: int,
    var2_n_categories: int,
    contingency_table: numpy.array,
    sigval: float,
) -> bool:
    """Computes the G-test for independence between two variables with checking
    for zeroes in the expectancy table"""
    # It's better to convert the entire count matrix to float for future calculations
    contingency_table = contingency_table.astype(numpy.float)

    expectancy_matrix = contingency.expected_freq(contingency_table)

    degrees_of_freedom = (var1_n_categories - 1) * (var2_n_categories - 1)

    division = numpy.divide(
        contingency_table,
        expectancy_matrix,
        out=numpy.zeros_like(contingency_table),
        where=expectancy_matrix != 0,
    )
    log_division = numpy.log(
        division, out=numpy.zeros_like(division), where=division != 0
    )
    the_sum = 2 * (contingency_table * log_division).sum()

    comparison = 1.0 - chi2.cdf(the_sum, degrees_of_freedom)
    return comparison >= sigval
