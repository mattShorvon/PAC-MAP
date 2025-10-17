from typing import Tuple, List, Optional
import copy
from spn.node.base import SPN
from spn.utils.evidence import Evidence
from spn.actions.map_algorithms.anytime_info import AnytimeInfo
from spn.structs import Variable


def local_search(spn: SPN, evidence: Evidence) -> AnytimeInfo:
    best_evidence = Evidence({var: [evidence[var][0]] for var in evidence.variables})
    best_value = spn.value(best_evidence)
    anytime_info = AnytimeInfo()
    anytime_info.new_lower_bound(best_value, best_evidence)
    for variable, values in evidence.items():
        for value in values:
            new_evidence = copy.deepcopy(best_evidence)
            new_evidence[variable] = [value]
            new_value = spn.value(new_evidence)
            if new_value > best_value:
                best_value = new_value
                best_evidence = new_evidence
                anytime_info.new_lower_bound(best_value, best_evidence)
    anytime_info.new_lower_bound(best_value, best_evidence)
    anytime_info.finish()
    return anytime_info


def local_search_with_evidence_and_marginalized(
    spn: SPN,
    evidence: Evidence,
    marginalized_variables: List[Variable],
    initial_evidence: Optional[Evidence] = None,
) -> AnytimeInfo:
    if initial_evidence is None:
        best_evidence = Evidence()
    else:
        best_evidence = initial_evidence
    for variable in spn.scope():
        if variable in marginalized_variables:
            best_evidence[variable] = list(range(variable.n_categories))
        elif variable in evidence and len(evidence[variable]) == 1:
            best_evidence[variable] = evidence[variable]
        elif variable not in evidence and variable not in best_evidence:
            best_evidence[variable] = [0]
    best_value = spn.value(best_evidence)
    anytime_info = AnytimeInfo()
    anytime_info.new_lower_bound(best_value, best_evidence)
    for variable, values in evidence.items():
        if variable in marginalized_variables or len(values) == 1:
            continue
        for value in values:
            new_evidence = copy.deepcopy(best_evidence)
            new_evidence[variable] = [value]
            new_value = spn.value(new_evidence)
            if new_value > best_value:
                best_value = new_value
                best_evidence = new_evidence
                anytime_info.new_lower_bound(best_value, best_evidence)
    anytime_info.new_lower_bound(best_value, best_evidence)
    anytime_info.finish()
    return anytime_info
