
from spn.actions.map_algorithms.merlin import merlin, make_uai_file
from spn.io.file import from_file
from pathlib import Path
from spn.structs import Variable
from spn.utils.evidence import Evidence


pathname = "test_inputs"
basename = "test_spn"

if pathname[-1] != "/":
    pathname = pathname + "/"

# Load SPN
spn = from_file(f"{pathname}{basename}/{basename}.spn")

# Convert the spn to a pgm
make_uai_file(spn, f"{pathname}{basename}/{basename}.uai")
sc = sorted(spn.scope())

# Load evidence
evidences = []
with open(f"{pathname}{basename}/{basename}.evid") as f:
    for line in f:
        line = line.split()
        if len(line) > 0 and int(line[0]) > 0:
            e = Evidence(
                {
                    sc[int(line[i])]: [int(line[i + 1])]
                    for i in range(1, 2 * int(line[0]) + 1, 2)
                }
            )
        else:
            e = Evidence()
        evidences.append(e)

# Load query
queries = []
with open(f"{pathname}{basename}/{basename}.query") as f:
    for line in f:
        line = line.split()
        q = [sc[int(line[i])] for i in range(1, int(line[0]) + 1)]
        queries.append(q)

# consistency check
assert len(evidences) == len(queries)

# Try out the merlin function
q_vars = [sc[0]]
result = merlin(
    spn = spn,
    evidence_file=f"{pathname}{basename}/{basename}.evid",
    query_file=f"{pathname}{basename}/{basename}.query",
    uai_file=f"{pathname}{basename}/{basename}.uai",
    ibound=10,
    iterations=2,
    query_vars=q_vars
)
print(result)
