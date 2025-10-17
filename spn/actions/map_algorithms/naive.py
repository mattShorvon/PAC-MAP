from typing import Tuple, List
from spn.utils.evidence import Evidence
from spn.node.base import SPN
from spn.structs import Variable


def naive(spn: SPN) -> Tuple[Evidence, float]:
    """Naive map will test all possible varsets on the SPN and select the one with the
    highest value"""
    return naive_with_evidence(spn, spn.all_marginalized())


def naive_with_evidence(spn: SPN, evidence: Evidence) -> Tuple[Evidence, float]:
    """Naive map restricted to a set of evidences highest value"""
    # total_number = evidence.map_problem_size()
    # current = 1
    best_value = -float("inf")
    best_evidence = None
    for ev in evidence.split():
        # print(f"Naive ({current}/{total_number})")
        value = spn.value(ev)
        if value > best_value:
            best_value = value
            best_evidence = ev
        # current += 1
    return best_evidence, best_value


def naive_with_evidence_and_marginals(spn: SPN, evidence: Evidence, marginals: List[Variable]) -> Tuple[Evidence, float]:
    """Restricted Naive Map with Evidence and Marginalized variables"""
    best_value = -float("inf")
    best_evidence = None
    for variable in spn.scope():
        if variable not in evidence:
            evidence[variable] = list(range(variable.n_categories))

    for ev in evidence.split_except(marginals):
        value = spn.value(ev)
        if value > best_value:
            best_value = value
            best_evidence = ev
    return best_evidence, best_value
