"""Article Script: Script to obtain the results displayed on the article:
Two Reformulation Approaches toMaximum-A-Posteriori Inference in Sum-Product Networks"""

import os
import sys
from pathlib import Path
from spn.io.file import from_file
from spn.actions.map_algorithms.max_product import (
    max_product_with_evidence_and_marginals,
)
from spn.actions.map_algorithms.local_search import (
    local_search_with_evidence_and_marginalized,
)
from spn.actions.map_algorithms.max_search import max_search, forward_checking
from spn.actions.map_algorithms.argmax_product import (
    argmax_product_with_evidence_and_marginalized,
)
from spn.utils.evidence import Evidence


def main():
    spns = []
    if len(sys.argv) < 2:
        print("Usage: python article_script.py SPN_FOLDER")
        sys.exit()
    prefix = sys.argv[1]
    for folder in os.scandir(prefix):
        if not folder.is_dir():
            continue
        dataset_name = folder.name
        spn_file = os.path.join(prefix, dataset_name, dataset_name + ".spn")
        if not os.path.exists(spn_file):
            continue
        spns.append((dataset_name, from_file(Path(spn_file))))

    spns = sorted(spns, key=lambda x: len(x[1].scope()))

    for dataset_name, spn in spns:
        query_filename = os.path.join(prefix, dataset_name, dataset_name + ".query")
        evid_filename = os.path.join(prefix, dataset_name, dataset_name + ".evid")
        print("#" * 80)
        print(f"SPN: {dataset_name}")
        print("#" * 80)
        queries = []
        evidences = []
        with open(query_filename, "r") as query_file:
            for line in query_file.readlines():
                var_ids = line.split(" ")[1:]
                query = [spn.scope()[int(var_id)] for var_id in var_ids]
                queries.append(query)
        with open(evid_filename, "r") as evid_file:
            for line in evid_file.readlines():
                evid_info = line.split(" ")[1:]
                index = 0
                evidence = Evidence()
                while index < len(evid_info):
                    var_id = int(evid_info[index])
                    value = int(evid_info[index + 1])
                    evidence[spn.scope()[var_id]] = [value]
                    index += 2
                evidences.append(evidence)
        for query, evidence in zip(queries, evidences):
            marginal_vars = [
                var for var in spn.scope() if var not in query and var not in evidence
            ]
            max_prod_evid, _ = max_product_with_evidence_and_marginals(
                spn, evidence, marginal_vars
            )
            max_prod_value = spn.value(max_prod_evid)
            local_search_evid = local_search_with_evidence_and_marginalized(
                spn, evidence, marginal_vars, max_prod_evid
            ).best_evidence()
            local_search_value = spn.value(local_search_evid)
            argmax_product_evid, _ = argmax_product_with_evidence_and_marginalized(
                spn, evidence, marginal_vars
            )
            argmax_product_value = spn.value(argmax_product_evid)
            max_search_evid, _ = max_search(
                spn,
                forward_checking,
                time_limit=600,
                marginalized_variables=marginal_vars,
                evidence=evidence,
            )
            max_search_value = spn.value(max_search_evid)
            print(
                f"""{dataset_name}, {max_prod_value},
                {local_search_value / max_prod_value},
                {argmax_product_value / max_prod_value},
                {max_search_value / max_prod_value}"""
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
