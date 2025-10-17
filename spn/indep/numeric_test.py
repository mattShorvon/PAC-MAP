"""Numeric Test: Contains independence test for two numeric random
variables using distance correlation."""
import numpy


def numeric_test(sample, sample2, threshold):
    """Test if two variable samples are independent using distance correlation."""
    corr = _correlation(sample.data, sample2.data)
    return corr < threshold


# pylint: disable=invalid-name
def _correlation(u, v):
    """
    This is a copy of scipy.spatial.distance.correlation with better handling of
    zero values.
    """
    umu = numpy.average(u)
    vmu = numpy.average(v)
    u = u - umu
    v = v - vmu
    uv = numpy.average(u * v)
    uu = numpy.average(numpy.square(u))
    vv = numpy.average(numpy.square(v))
    if uu * vv == 0:
        return 0.0

    dist = 1.0 - uv / numpy.sqrt(uu * vv)
    return dist
