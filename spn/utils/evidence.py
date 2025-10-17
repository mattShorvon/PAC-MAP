"""Evidence can also be called Varset, or variable assignment. We're representing
both the varsets with single variable assignments and partial evidences (with
multiple assignments representing multiple varsets)"""

from typing import List, Dict, Union, Optional, Generator
from collections.abc import ValuesView, KeysView
import itertools
import functools
import operator

from spn.structs import Variable

ContinuousVarset = Dict[Variable, List[float]]
DiscreteVarset = Dict[Variable, List[int]]

VarsetType = Union[ContinuousVarset, DiscreteVarset]


class Evidence:
    """Holds the dict that represents this assignment."""

    def __init__(self, varset: Optional[VarsetType] = None):
        if varset is None:
            self.__varset: VarsetType = {}
        else:
            self.__varset = varset

    def __repr__(self):
        string_representation = ""
        for var in self.variables:
            string_representation += "{} => {} ".format(var.id, self[var])
        return string_representation

    def __hash__(self):
        to_hash = {k: frozenset(v) for k, v in self.__varset.items()}
        return hash(frozenset(to_hash))

    def __getitem__(self, key: Variable):
        return self.__varset[key]

    def __setitem__(self, key: Variable, value: Union[List[float], List[int]]):
        self.__varset[key] = value

    def __eq__(self, other: object):
        if not isinstance(other, Evidence):
            return NotImplemented
        if self.__varset.keys() == other.varset.keys():
            for key, value in other.varset.items():
                if self.__varset[key] != value:
                    return False
            return True
        return False

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.__varset)

    def __contains__(self, item):
        return item in self.__varset

    def has_subevidence(self, subevidence: "Evidence") -> bool:
        """Returns True if self represents a set of evidences that contains the
        subevidence"""
        for variable, values in subevidence.items():
            for value in values:
                if value not in self[variable]:
                    return False
        return True

    def all_set(self):
        """Returns true if all variables have exactly one assignment"""
        return all([len(assignment) == 1 for assignment in self.__varset.values()])

    @property
    def varset(self) -> Dict[Variable, Union[List[float], List[int]]]:
        """The underlying varset Dict"""
        return self.__varset

    @property
    def variables(self) -> List[Variable]:
        """Returns the list of variables in this evidence sorted by id"""
        return sorted(self.keys(), key=lambda x: x.id)

    def has_var(self, var: Variable) -> bool:
        """Returns True if the underlying dict has the var key"""
        return var in self.__varset

    def value(self, var: Variable) -> Union[List[float], List[int]]:
        """Returns the value of the assignment at the parameter var_id"""
        return self.__varset[var]

    def items(self):
        """Returns an iterator over the variable ids and values, like a dict"""
        return self.__varset.items()

    def union(self, evidence: "Evidence"):
        """Adds the values from the evidence parameter into the values of the current
        varset"""
        for key, values in evidence.varset.items():
            for value in values:
                if value not in self.__varset[key]:
                    self.__varset[key].append(value)

    def merge(self, evidence: "Evidence"):
        """Adds the values and keys from the evidence parameter into the current
        varset"""
        for key, values in evidence.varset.items():
            if key not in self.__varset:
                self.__varset[key] = values
            else:
                for value in values:
                    if value not in self.__varset[key]:
                        self.__varset[key].append(value)

    def keys(self) -> KeysView:
        """Returns a list with the variables in this Evidence"""
        return self.__varset.keys()

    def values(self) -> ValuesView:
        """Returns a list with the values for each variable in this Evidence"""
        return self.__varset.values()

    def split(self) -> Generator["Evidence", None, None]:
        """Returns a list of Evidences, each with one value for each variable
        obtained from the list of values in this current evidence"""
        variables = self.variables
        lists_of_values = [self[var] for var in variables]
        combinations = itertools.product(*lists_of_values)
        return (
            Evidence(
                {variable: [value] for variable, value in zip(variables, combination)}
            )
            for combination in combinations
        )
    
    def split_except(self, except_vars: List[Variable]) -> Generator["Evidence", None, None]:
        """Returns a list of Evidences, each with one value for each variable
        obtained from the list of values in this current evidence, with the exception of the list of variables passed"""
        variables = [var for var in self.variables if var not in except_vars]
        lists_of_values = [self[var] for var in variables]
        combinations = itertools.product(*lists_of_values)
        return (
            Evidence(
                {variable: [value] for variable, value in zip(variables, combination)}
            )
            for combination in combinations
        )

    def change_value_at_id(self, var_id: int, value: List[int]):
        """Changes the value at the variable with the id var_id"""
        for variable in self.variables:
            if variable.id == var_id:
                self[variable] = value
                return

    def map_problem_size(self) -> int:
        """Returns the size of the solution set of a map_problem restricted
        to only the variables in this evidence"""
        return functools.reduce(
            operator.mul, [len(values) for values in self.__varset.values()]
        )

    @staticmethod
    def from_data(data: List[int], variables: List[Variable]):
        """Returns an evidence created with an array of data. The variable ids
        are used to index into the data list."""
        varset = {variable: [data[variable.id]] for variable in variables}
        return Evidence(varset)
