"""Graph: Contains the IndependencyGraph class for the comparison of datasets
from multiple variables to detect independency, and an implementation of the
g-test for independence"""

import logging
from typing import List, Mapping
from spn.indep.unionfind import UnionFindNode, node_union
from spn.indep.categorical_test import categorical_test
from spn.indep.numeric_test import numeric_test
from spn.structs import Vardata


class IndependencyGraph:
    """Graph that uses Union Find for detecting dependencies among variables"""

    def __init__(self, var_datas: List[Vardata], pval: float, numeric: bool):
        self.__adjacency_list = {}
        ids = {}
        for sample_index, sample in enumerate(var_datas):
            ids[sample_index] = sample.variable.id
            self.__adjacency_list[ids[sample_index]] = []

        logging.info("Constructing independence graph...")

        # Sets of Union Find Trees
        sets = [UnionFindNode(ids[i]) for i in range(len(var_datas))]

        for sample_index, sample in enumerate(var_datas):
            for sample2_index, sample2 in enumerate(var_datas[1:]):
                sample_var_id, sample2_var_id = ids[sample_index], ids[sample2_index]

                if sets[sample_index].find() == sets[sample2_index].find():
                    continue

                if numeric:
                    indep = numeric_test(sample, sample2, 0.5)
                else:
                    indep = categorical_test(sample, sample2, pval)

                if not indep:
                    self.__adjacency_list[sample_var_id].append(sample2_var_id)
                    self.__adjacency_list[sample2_var_id].append(sample_var_id)
                    node_union(sets[sample_index], sets[sample2_index])

        self.__kset = []
        for var_set in sets:
            if var_set == var_set.parent:
                self.__kset.append(var_set.var_ids())


    @property
    def adjacency_list(self) -> Mapping[int, List[int]]:
        """Adjacency list as a matrix of variable ids that were detected as
        adjacent on the graph, and therefore dependent"""
        return self.__adjacency_list

    @property
    def kset(self) -> List[List[int]]:
        """The lists of variable ids obtained from the independent subgraphs"""
        return self.__kset
