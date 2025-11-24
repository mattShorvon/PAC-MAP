from spn.io.file import from_file
from spn.actions.map_algorithms.pac_map import pac_map
from spn.utils.evidence import Evidence
from pathlib import Path

# Load an SPN
spn = from_file(Path("test_inputs/iris/iris.spn"))
evid = Evidence()
evid[spn.scope()[0]] = [1]
test = spn.value(evid)
marg_vars = [spn.scope()[3], spn.scope()[4]]
pac_map(spn, evid, marg_vars)
