"""Varset Utils: Utility functions for dicts of variables and values"""

import copy
from typing import List, Dict, Union
from spn.structs import Variable
from spn.utils.evidence import Evidence


def possible_values(scope: List[Variable]) -> List[Dict[Variable, int]]:
    """Returns all the possible varsets that can be
    made with the variables in the scope"""
    variable = scope[0]
    varsets = []
    for value in range(variable.n_categories):
        varsets.append({variable: value})

    if len(scope) == 1:
        return varsets

    new_varsets = []
    varsets_from_other_variables = possible_values(scope[1:])
    for varset in varsets:
        new_varset = {}
        for varset_from_other_variables in varsets_from_other_variables:
            for var_id, value in varset.items():
                new_varset[var_id] = value
            for var_id, value in varset_from_other_variables.items():
                new_varset[var_id] = value
            new_varsets.append(copy.deepcopy(new_varset))
    return new_varsets


def varset_union(varsets: List[Dict[Variable, Union[List[int], int]]]) -> Dict[Variable, List[int]]:
    """Combines varsets of the type var_id -> Value into a varset
    of type var_id -> List[Value]"""
    final_varset: Dict[Variable, List[int]] = {}
    for varset in varsets:
        for key, values in varset.items():
            if isinstance(values, int):
                values = [values]
            if key in final_varset:
                for value in values:
                    if value not in final_varset[key]:
                        final_varset[key].append(value)
            else:
                final_varset[key] = values
    return final_varset


def possible_complete_evidences(scope: List[Variable]) -> List[Evidence]:
    """Returns all the possible complete evidences given a list of variables
    Complete evidences are maps from variable ids to
    lists with only one assignment for each variable"""
    # Type checking doesn't work because of multiple unwrappings of Optional values
    return [Evidence(varset) for varset in possible_values(scope)] # type: ignore


def all_variables_marginalized(scope: List[Variable]) -> Evidence:
    """Returns an Evidence representing a varset with all variables marginalized. In
    an SPN, setting all indicators to 1 for a variable will allow for the
    calculation of marginals"""
    varset = {variable: list(range(variable.n_categories)) for variable in scope}
    # Type checking doesn't work because of multiple unwrappings of Optional values
    return Evidence(varset) # type: ignore
