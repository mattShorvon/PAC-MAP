
import copy

from spn.node.base import SPN
from spn.utils.evidence import Evidence

def beam_search(spn: SPN, beam_size: int) -> Evidence:
    """Beam search works much like a hill climbing algorithm, with a size restricting
    the search space. Starting from an initial sample (all assigned to zero), it
    then changes one variable at a time and keeps the [beam_size] best samples.

    From: Maximum A Posteriori Inference in Sum-Product Networks (2017)"""
    variables = spn.scope()
    initial_evidence = Evidence({variable: [0] for variable in variables})
    current_evidences = [(initial_evidence, spn.value(initial_evidence))]
    while True:
        better_evidences = []
        for evidence, value in current_evidences:
            derivatives = spn.derivative_of_assignment(evidence)
            # Find the changes in assignment that will increase the value
            for variable_id, new_values in derivatives.items():
                for new_value_index, new_value in enumerate(new_values):
                    if new_value > value:
                        new_evidence = copy.deepcopy(evidence)
                        new_evidence.change_value_at_id(variable_id, [new_value_index])
                        better_evidences.append((new_evidence, new_value))
        if not better_evidences:
            return current_evidences[0][0]
        better_evidences = sorted(
            better_evidences,
            key=lambda evidence_and_value: evidence_and_value[1],
            reverse=True
        )[:beam_size]
        current_evidences = better_evidences
