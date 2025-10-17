from typing import Tuple, Dict, Optional
from spn.utils.evidence import Evidence
from spn.structs import Variable
from spn.actions.map_algorithms.hybrid_max_product import hybrid_max_product
from spn.node.base import SPN


class BoundState:
    """Stores the state of the branching: The upper and lower bounds, and the best
    evidence found so far"""

    def __init__(self, lower: float, upper: float, evidence: Optional[Evidence]):
        self.lower = lower
        self.upper = upper
        self.evidence = evidence

    def __repr__(self):
        return f"{self.lower} - {self.upper} - {self.evidence}"


def get_default_bound_state(
    spn: SPN, evidence: Evidence
) -> Tuple[BoundState, Dict[Variable, Dict[int, float]]]:
    max_prod_evidence, _, upper_bound_value, marginal_dict = hybrid_max_product(
        spn, evidence
    )
    if max_prod_evidence is None:
        return BoundState(0, 0, evidence), marginal_dict
    for var, values in evidence.items():
        if var not in max_prod_evidence.variables:
            if len(values) > 1:
                raise ValueError(
                    f"Hybrid max product returned empty assignment for variable {var} but the evidence set has the values {values}"
                )
            max_prod_evidence[var] = values

    lower_bound_value = (
        spn.value(max_prod_evidence) if max_prod_evidence is not None else 0.0
    )
    return (
        BoundState(lower_bound_value, upper_bound_value, max_prod_evidence),
        marginal_dict,
    )
