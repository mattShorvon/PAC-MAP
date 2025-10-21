from spn.node.base import SPN
from spn.node.indicator import Indicator
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from typing import Callable, cast, Optional, Tuple, List, Dict
from spn.utils.evidence import Evidence
from spn.structs import Variable
from functools import reduce


# helper functions
def argmax(vector:list):
    " Returns the position of the maximum value "
    i = 0
    for j in range(len(vector)):
        if vector[i] < vector[j]:
            i = j
    return i

def product(vector:list) -> float:
    " Computes the product of the numbers in list vector"
    return reduce(lambda x,y: x*y, vector)


# Message passing algorithm
def lbp(spn, evidence, marginalized=[], num_iterations=100, tolerance=1e-6):
    " Implements Liu & Ihler 2011's max-sum-product belief propagation algorithm "
    sc = sorted(spn.scope())
    print(f"{len(sc)} variables: {len(sc)-len(evidence)-len(marginalized)} query, {len(evidence)} evidence, {len(marginalized)} marginalized.")
    pi: Dict[(SPN,SPN),float] = {} # downstream messages
    pi_i: Dict[(int,SPN),List[float]] = {} # downstream messages from root (max) node
    ll: Dict[(SPN,SPN),float] = {} # upstream messages
    ll_i: Dict[(SPN,int),List[float]] = { } # upstream messages to root (max) nodes
    parent: Dict[SPN,List[SPN]] = {} # parent of each node
    parent_i: Dict[int,List[SPN]] = {} # parent of root nodes
    bel: Dict[int,List[float]] = {} # beliefs for root nodes
    num_sum, num_prod, num_leaves = 0, 0, 0
    prev_value = 0.0
    ### Initialization ################################################################
    for v in sc:
        bel[v.id] = [1/v.n_categories for _ in range(v.n_categories)]
    for node in reversed(spn.topological_order()):
        assert len(node.children) <= 2
        if node.type == "leaf":
            num_leaves += 1
            node = cast(Indicator,node)
            ncat = node.variable.n_categories
            ll_i[(node,node.variable.id)] = [1.0/ncat for _ in range(ncat)]
            if (node.variable.id,node) not in pi_i:
                pi_i[(node.variable.id,node)] = [1.0/ncat for _ in range(ncat)]
            if node.variable.id in parent_i:
                parent_i[node.variable.id].append(node)
            else:
                parent_i[node.variable.id] = [node]
        else:
            if node.type == "sum":
                #node = cast(SumNode,node)
                num_sum += 1
            elif node.type == "product":
                #node = cast(ProductNode,node)
                num_prod += 1
            else:
                raise Exception("Unknown node type")
            for child in node.children:
                ll[(node,child)] = 0.5
                pi[(child,node)] = 0.5
                if child in parent:
                    parent[child].append(node)
                else:
                    parent[child] = [ node ]
    # ensure there's one leaf for each value of each variable
    print(f"SPN has {num_sum} sums, {num_prod} products and {num_leaves} leaves.\n")
    assert num_sum+num_prod+num_leaves == len(spn.topological_order()) 
    assert num_leaves == sum(v.n_categories for v in spn.scope())
    print("iteration | value ")
    for iteration in range(num_iterations):
        ### Downward propagation ##########################################################
        # compute message from root node v to child (indicator) node
        for (v,node) in pi_i:
            # TODO handle evidence
            values = pi_i[(v,node)]
            if sc[v] in evidence: 
                # print(sc[v])
                ve = evidence[sc[v]][0]
                for j in range(sc[v].n_categories):
                        values[j] == 0.0
                values[ve] = 1.0
            else:
                Z = 0.0
                for j in range(node.variable.n_categories):
                    values[j] = product(ll_i[(n,v)][j] for n in parent_i[v] if n != node)
                    Z += values[j]
                # normalize messages (seems to be important only for marginalized variables)
                for j in range(node.variable.n_categories):
                    values[j] /= Z 
            #print("root",v,node.assignment,values,Z)
        # compute remaining messages  
        for node in reversed(spn.topological_order()):
            # compute messages pi_X^Y -- assumes incoming messages are normalized
             if node.type == "leaf":
                vid = node.variable.id
                maxbel = max(bel[vid]) 
                for pa in parent[node]:
                    if node.variable in marginalized:

                        pi1 = pi_i[(vid,node)][node.assignment]*product(ll[(opa,node)] for opa in parent[node] if opa != pa)
                        pi0 = (1-pi_i[(vid,node)][node.assignment])*product(1.0-ll[(opa,node)] for opa in parent[node] if opa != pa)
                    else:
                        # TODO: handle evidence more efficiently
                        if bel[vid][node.assignment] == maxbel:
                            pi1 = pi_i[(vid,node)][node.assignment]*product(ll[(opa,node)] for opa in parent[node] if opa != pa)
                        else:
                            pi1 = 0.0
                        pi0 = 0.0
                        for j in range(node.variable.n_categories):
                            if j != node.assignment:
                                if bel[vid][j] == maxbel:
                                    pi0 += pi_i[(vid,node)][j]
                        pi0 *= product(1.0-ll[(opa,node)] for opa in parent[node] if opa != pa)
                    pi[(node,pa)] = pi1/(pi1+pi0)  # normalization
                    #print("leaf", vid, node.assignment, pi[(node,pa)], pi1, pi0, pi_i[vid,node])
             elif node.type == "sum":
                 if not(node.root):
                     node = cast(SumNode, node)
                     assert len(parent[node]) == 1
                     pa = parent[node][0]
                     pi[(node,pa)] = node.weights[0]*pi[(node.children[0],node)]+node.weights[1]*pi[(node.children[1],node)]
             else: # Product
                 if not(node.root):
                     assert len(parent[node]) == 1
                     pa = parent[node][0]
                     pi[(node,pa)] = pi[(node.children[0],node)]*pi[(node.children[1],node)] 

        ### Upward propagation ##########################################################
        for node in spn.topological_order():
            # compute messages ll_X^U -- assumes incoming messages are normalized
             if node.type == "sum":
                 node = cast(SumNode,node)
                 ch1,ch2 = node.children
                 if spn.root: # root note receives indicator on value = 1 (to represent evidence on that node)
                     llp = 1.0
                 else:
                     assert len(parent[node]) == 1
                     pa = parent[node][0]
                     llp = ll[(pa,node)]
                 # left child
                 ll1 = (1-llp)*node.weights[1]*(1-pi[(ch2,node)]) + llp*(node.weights[0]*(1-pi[(ch2,node)])+pi[(ch2,node)])
                 ll0 = (1-llp)*(1-pi[(ch2,node)] + node.weights[0]*pi[(ch2,node)]) + llp*node.weights[1]*pi[(ch2,node)]
                 ll[(node,ch1)] = ll1/(ll0+ll1)
                 # right child
                 ll1 = (1-llp)*node.weights[0]*(1-pi[(ch1,node)]) + llp*(node.weights[1]*(1-pi[(ch1,node)])+pi[(ch1,node)])
                 ll0 = (1-llp)*(1-pi[(ch1,node)] + node.weights[1]*pi[(ch1,node)]) + llp*node.weights[0]*pi[(ch1,node)]
                 ll[(node,ch2)] = ll1/(ll0+ll1)
             elif node.type == "product": 
                 ch1,ch2 = node.children
                 if spn.root:
                     llp = 1.0
                 else:
                     assert len(parent[node]) == 1
                     pa = parent[node][0]
                     llp = ll[(pa,node)]
                 # left child
                 ll1 = (1.0-llp)*(1.0-pi[(ch2,node)]) + llp*pi[(ch2,node)]
                 ll[(node,ch1)] = ll1/(ll1+1-llp)
                 # right child
                 ll1 = (1.0-llp)*(1.0-pi[(ch1,node)]) + llp*pi[(ch1,node)]
                 ll[(node,ch2)] = ll1/(ll1+1-llp)
        # compute messages to root nodes
        for (n,v) in ll_i:
            llp1 = product(ll[(pa,n)] for pa in parent[n])
            llp0 = product(1.0-ll[(pa,n)] for pa in parent[n])
            Z = llp1+llp0*(sc[v].n_categories-1)
            for j in range(sc[v].n_categories):
                ll_i[(n,v)][j] = llp0/Z
            ll_i[(n,v)][n.assignment] = llp1/Z
        ### Compute beliefs ##########################################################
        for v in sc:
            if v in evidence:
                ve = evidence[v][0]
                for j in range(v.n_categories):
                        bel[v.id][j] == 0
                bel[v.id][ve] = 1.0
            else:
                Z = 0.0
                for j in range(v.n_categories):
                    b = product(ll_i[(n,v.id)][j] for n in parent_i[v.id])
                    Z += b
                    bel[v.id][j] = b
                for j in range(v.n_categories):
                    bel[v.id][j] /= Z
        #print(bel)
        # x = Evidence(dict((v,[argmax(bel[v.id])]) for v in sc))
        # print(x, spn.value(x))
        # extract approximated value (the marginal of the leaf) 
        if spn.type == "sum": # root node is sum
            value = spn.weights[0]*pi[(spn.children[0],spn)]+spn.weights[1]*pi[(spn.children[1],spn)]
            print(f"{iteration:9} | {value}")
            if abs(value-prev_value) < tolerance:
                # early stop
                break
            prev_value = value

    x = Evidence(dict((v,[argmax(bel[v.id])]) for v in sc))
    for v in marginalized:
        x[v] = [i for i in range(v.n_categories)]
    return x

if __name__ == "__main__":
    from spn.io.file import to_file, from_file
    from spn.actions.map_algorithms.max_product import *
    from spn.actions.map_algorithms.max_search import * 
    from spn.utils.graph import full_binarization
    import sys
    pathname = sys.argv[1]

    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], " pathname basename\n  Example:", sys.argv[0],"~/learned-spns/ ionoshpere")
        exit(0)

    run_mp = True # run MaxProduct?
    run_ms = False # run MaxSearh?

    if pathname[-1] != "/":
        pathname = pathname + "/"
    basename = sys.argv[2]
    # Load SPN
    spn = from_file(f"{pathname}{basename}/{basename}.spn")
    print(f"SPN has {len(spn.topological_order())} nodes, {len(spn.scope())} variables.")
    spn = full_binarization(spn)
    spn.fix_scope()
    spn.fix_topological_order()
    print(f"Binarized SPN has {len(spn.topological_order())} nodes, {len(spn.scope())} variables.")
    print()

    sc = sorted(spn.scope())
    #print("Scope:", sc)

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
        # for node in bel:
        #     if node.root:
        #         print('*', end = ' ')
        #     if node.type == "leaf":
        #         print(node.type, node.variable.id, node.assignment, bel[node])
        #     else:
        #         print(node.type, bel[node])

