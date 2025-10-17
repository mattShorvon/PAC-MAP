"""Structs: Contains the common types used in various functions that do not have
enough operations to be considered classes"""

from typing import NamedTuple
import numpy

# The id is also the index to be searched at the data samples, as if the data
# samples were dicts in which the keys are the variable ids
class Variable(NamedTuple):
    id: int
    n_categories: int


class Vardata(NamedTuple):
    variable: Variable
    data: numpy.array
