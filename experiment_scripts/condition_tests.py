import numpy as np
import pandas as pd
from spn.learn import gens
from pathlib import Path
from spn.data.partitioned_data import PartitionedData
from spn.io.file import to_file, from_file
from spn.actions.sample import sample_multiproc, sample
from spn.actions.likelihood import likelihood_multiproc
from spn.actions.condition import condition_spn
from spn.utils.evidence import Evidence
import os
import time

# Testing the condition_spn() function to make sure it pasts a few sense checks
# The probabilities it outputs should be the same as the ones you get from 
# querying the original spn
if __name__ == "__main__":
    # Toy data
    spn_path = Path('test_inputs/toy_data/toy_data.spn')
    spn = from_file(spn_path)
    var_x1 = spn.scope()[0]
    var_x2 = spn.scope()[1]
    evid = Evidence({var_x1: [1]})
    spn_conditioned = condition_spn(spn, evidence=evid)
    timestamp = int(time.time())
    conditioned_spn_path = spn_path.parent / f"{spn_path.stem}_conditioned_{timestamp}_{os.getpid()}.spn"
    to_file(spn_conditioned, conditioned_spn_path)
    print("Toy data and spn loaded, spn conditioned")
    p_evid = spn.value(evid)
    query = Evidence({var_x1: [1], var_x2: [1]})
    p_original_spn = np.exp(
        likelihood_multiproc(spn_path, [query], n_jobs=1)
    ) / p_evid
    p_conditioned_spn = np.exp(
        likelihood_multiproc(conditioned_spn_path, [query], n_jobs=1)
    )
    print(f"Original spn: {p_original_spn}")
    print(f"Conditioned spn: {p_conditioned_spn}")
    conditioned_spn_path.unlink()

    # Medium data
    spn_path = Path('20-datasets/mushrooms/mushrooms.spn')
    spn = from_file(spn_path)
    var_x1 = spn.scope()[0]
    var_x2 = spn.scope()[1]
    evid = Evidence({var_x1: [1]})
    spn_conditioned = condition_spn(spn, evidence=evid)
    timestamp = int(time.time())
    conditioned_spn_path = spn_path.parent / f"{spn_path.stem}_conditioned_{timestamp}_{os.getpid()}.spn"
    to_file(spn_conditioned, conditioned_spn_path)
    print("Mushrooms data and spn loaded, spn conditioned")
    p_evid = spn.value(evid)
    query = Evidence({var_x1: [1], var_x2: [1]})
    p_original_spn = np.exp(
        likelihood_multiproc(spn_path, [query], n_jobs=1)
    ) / p_evid
    p_conditioned_spn = np.exp(
        likelihood_multiproc(conditioned_spn_path, [query], n_jobs=1)
    )
    print(f"Original spn: {p_original_spn}")
    print(f"Conditioned spn: {p_conditioned_spn}")
    conditioned_spn_path.unlink()

    # Large data
    spn_path = Path('20-datasets/cwebkb/cwebkb.spn')
    spn = from_file(spn_path)
    var_x1 = spn.scope()[0]
    var_x2 = spn.scope()[1]
    evid = Evidence({var_x1: [1]})
    spn_conditioned = condition_spn(spn, evidence=evid)
    timestamp = int(time.time())
    conditioned_spn_path = spn_path.parent / f"{spn_path.stem}_conditioned_{timestamp}_{os.getpid()}.spn"
    to_file(spn_conditioned, conditioned_spn_path)
    print("Cwebkb data and spn loaded, spn conditioned")
    p_evid = spn.value(evid)
    query = Evidence({var_x1: [1], var_x2: [1]})
    p_original_spn = np.exp(
        likelihood_multiproc(spn_path, [query], n_jobs=1)
    ) / p_evid
    p_conditioned_spn = np.exp(
        likelihood_multiproc(conditioned_spn_path, [query], n_jobs=1)
    )
    print(f"Original spn: {p_original_spn}")
    print(f"Conditioned spn: {p_conditioned_spn}")
    conditioned_spn_path.unlink()
