import sys

if len(sys.argv) < 3:
    print(
        "Usage:",
        sys.argv[0],
        " pathname basename\n  Example:",
        sys.argv[0],
        "~/learned-spns/ ionoshpere",
    )
    exit(0)


from spn.actions.map_algorithms.merlin import *
from spn.io.file import to_file, from_file
from spn.actions.map_algorithms.max_product import *
from spn.actions.map_algorithms.local_search import *
from spn.actions.map_algorithms.argmax_product import *
from spn.actions.map_algorithms.max_search import *
from spn.utils.graph import full_binarization
from experiment_scripts.lbp import lbp

from os import path

pathname = sys.argv[1]
if pathname[-1] != "/":
    pathname = pathname + "/"
basename = sys.argv[2]
# Load SPN
spn = from_file(f"{pathname}{basename}/{basename}.spn")
# Binarize SPN
spn = full_binarization(spn)
spn.fix_scope()
spn.fix_topological_order()
print(f"Binarized SPN has {len(spn.topological_order())} nodes, {len(spn.scope())} variables.")

# Output fixed uai file
#make_uai_file(spn, f"{pathname}{basename}/{basename}.uai")

sc = sorted(spn.scope())

# Load evidence
evidences = []
with open(f"{pathname}{basename}/{basename}.evid") as f:
    for line in f:
        line = line.split()
        if len(line) > 0:
            if int(line[0]) > 0:
                e = Evidence(
                    {
                        sc[int(line[i])]: [int(line[i + 1])]
                    for i in range(1, 2 * int(line[0]) + 1, 2)
                    }
                )
                evidences.append(e)
            else:
                evidences.append(Evidence())

# Load query
queries = []
with open(f"{pathname}{basename}/{basename}.query") as f:
    for line in f:
        line = line.split()
        if len(line) > 0:
            if int(line[0]) > 0:
                q = [sc[int(line[i])] for i in range(1, int(line[0]) + 1)]
                queries.append(q)
            else:
                queries.append([])

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
    x_mp = max_product_with_evidence_and_marginals(spn, e, m)[0]
    v_mp = spn.value(x_mp)
    print("Max-Product:", v_mp)
    x_ls = local_search_with_evidence_and_marginalized(
        spn, e, m, initial_evidence=x_mp
    ).best_evidence()
    v_ls = spn.value(x_ls)
    print("Max-Product+Local-search:", v_ls)
    x_ap = argmax_product_with_evidence_and_marginalized(spn, e, m)[0]
    v_ap = spn.value(x_ap)
    print("ArgMax-Product:", v_ap)
    x_ms = max_search(spn, forward_checking, evidence=e, marginalized_variables=m)[0]
    v_ms = spn.value(x_ms)
    #v_ms = 0
    print("Max-Search:", v_ms)
    v_wmb = 0
    try:
        x_bp = lbp(spn, e, m)
        v_bp = spn.value(x_bp)
        print("Belief-Prop:", v_bp)
    except:
        v_bp = 0
        print("Belief-Prop: numerical error")

    v_wmb = 0
    if path.exists(f"{pathname}{basename}/wmb{i+1}.MMAP"):
        # Load configuration
        x_wmb = x_mp  # so that the marginalized variables and evidence is set properly
        with open(f"{pathname}{basename}/wmb{i+1}.MMAP") as f:
            line = f.readline()  # ignore header
            line = f.readline().split()
            sq = sorted([v.id for v in q])
            for i in range(1, int(line[0]) + 1):
                value = int(line[i])
                x_wmb[sc[sq[i - 1]]] = [value]
                v_wmb = spn.value(x_wmb)
        print("WMB:", v_wmb)
    print()
    results += f"& {v_mp:.2e} & {v_ls/v_mp:.3f} & {v_ap/v_mp:.3f} & {v_ms/v_mp:.3f} & {v_wmb/v_mp:.3f} & & {v_bp/v_mp:.3f} \\\ \n"
print("### RESULTS #################################")
print("  MP         MP+LS   AMP     MS      WMB       HBP   ")
print(results)


# use
# ~/merlin/bin/merlin -a wmb -i 10 -n 30 -t MMAP --input-file "${basename}.uai" --query-file "${basename}.query" --evidence-file "${basename}.evid" --output-format uai --output-file "wmb" --threshold 1e-8 -l 600

# AAOBF
#  ~/mmap-solver/bin/mmap -a any-aaobf --input-file "${basename}.uai" -M "${basename}.query" --evidence-file "{basename}.evid" -F uai
# v_aaobf = 0
# if path.exists(f"{pathname}{basename}/{basename}.map"):
#     # Load configuration
#     x_aaobf = x_mp # so that the marginalized variables and evidence is set properly
#     empty = True
#     with open(f"{pathname}{basename}/{basename}.map") as f:
#         lines = f.readlines() # ignore header
#         if len(lines) > 0:
#             empty = False
#             line = lines[-1].split()
#             for i in range(1,int(line[0])+1):
#                 value = int(line[i])
#                 x_aaobf[q[i-1]] = [value]
#     if not(empty):
#         v_aaobf = spn.value(x_aaobf)
#         print("AAOBF:", v_aaobf)

