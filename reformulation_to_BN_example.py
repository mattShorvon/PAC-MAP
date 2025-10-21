from spn.actions.map_algorithms.merlin import merlin, make_uai_file
from spn.io.file import from_file
from pathlib import Path
from spn.structs import Variable
from lbp import lbp
from spn.utils.graph import full_binarization
import sys
from spn.utils.evidence import Evidence


pathname = "test_inputs"
basename = "iris"

run_mp = False # run MaxProduct?
run_ms = False # run MaxSearh?

if pathname[-1] != "/":
    pathname = pathname + "/"

# Load SPN
spn = from_file(f"{pathname}{basename}/{basename}.spn")
print(f"SPN has {len(spn.topological_order())} nodes, {len(spn.scope())} variables.")
spn = full_binarization(spn)
spn.fix_scope()
spn.fix_topological_order()
print(f"Binarized SPN has {len(spn.topological_order())} nodes, {len(spn.scope())} variables.")
print()

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

print("#Cases:    \t", len(queries))

# Compute marginalized vars
results = ""
for i in range(len(evidences)):
    print("### CASE", i, "##############################")
    e, q = evidences[i], queries[i]
    m = [v for v in sc if v not in e and v not in q]
    # sanity check
    assert len(e) + len(m) + len(q) == len(sc)
    print("Evidence    :", e)
    print("Query       :", ' '.join([f"{v.id}({v.n_categories})" for v in q ]))
    print("Marginalized:", ' '.join([f"{v.id}({v.n_categories})" for v in m ]))
    print()

    if run_mp:
        x_mp = max_product_with_evidence_and_marginals(spn, e, m)[0]
        v_mp = spn.value(x_mp)
        print(f"MP:           {v_mp:.4e}")
        print("Configuration:", ' '.join([str(x_mp[v]) for v in q]))
        print()

    if run_ms:
        x_ms = max_search(spn,forward_checking,evidence=e,marginalized_variables=m)
        v_ms = spn.value(x_ms)
        print(f"MS:           {v_ms:.4e}    {v_ms/v_mp:.3f}")
        print("Configuration:", ' '.join([str(x_ms[v]) for v in q]))
        print()

    x_bp = lbp(spn, e, m, num_iterations=5)
    v_bp = spn.value(x_bp)
    print()
    if run_mp:
        print(f"BP:           {v_bp:.4e}    {v_bp/v_mp:.3f}")
    else:
        print(f"BP:           {v_bp:.4e}")
    print("Configuration:", ' '.join([str(x_bp[v]) for v in q]))
    print()



# # Load the spn you learned using LearnSPN & app.py + config.json
# spn_file = "test_inputs/test_spn/test_spn.spn"
# spn_test = from_file(Path(spn_file))

# # Convert the spn to a pgm
# make_uai_file(spn_test, "test_inputs/test_spn/test_spn.uai")



# # Try out the merlin function
# q_vars = [Variable(0, 1), Variable(1, 1), Variable(2, 1), Variable(3, 1),
#               Variable(4, 1)]
# result = merlin(
#     spn = spn_test,
#     evidence_file="test_inputs/test_spn/test_spn.evid",
#     query_file="test_inputs/test_spn/test_spn.query",
#     uai_file="test_inputs/test_spn/test_spn.uai",
#     ibound=10,
#     iterations=2,
#     query_vars=q_vars
# )
# print(result)
