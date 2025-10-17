"MIS: Contains the interface for manipulating MIS instances"

import math
import random
import numpy as np
from spn.graph.base import Base
from spn.node.base import SPN
from spn.node.multinomial import MultinomialNode
from spn.node.product import ProductNode
from spn.node.sum import SumNode
from spn.structs import Variable


class MIS(Base):
    """A MIS-instance. The problem of Minimum independent set, as per the description in
    http://sites.nlsde.buaa.edu.cn/~kexu/benchmarks/graph-benchmarks.htm"""

    def __init__(self, n: int, alpha: float, r: float, p: float):
        assert alpha > 0
        assert 0 < p < 1
        assert r > 0
        vertices_per_clique = int(n ** alpha)
        n_vertices = n * vertices_per_clique
        bool_matrix = np.zeros([n_vertices, n_vertices])
        last_vertice = 0
        n_edges = n * (vertices_per_clique * (vertices_per_clique - 1) / 2)
        for _ in range(n):
            for j in range(vertices_per_clique):
                for k in range(vertices_per_clique):
                    bool_matrix[last_vertice + j][last_vertice + k] = True
            last_vertice += vertices_per_clique

        for _ in range(int(r * n * math.log(n)) - 1):
            clique_1, clique_2 = np.random.choice(n, 2, replace=False)
            for _ in range(int(p * (n ** (2 * alpha)))):
                vertices_1 = list(range(clique_1, clique_1 + vertices_per_clique))
                vertices_2 = list(range(clique_2, clique_2 + vertices_per_clique))
                choice_1 = random.choice(vertices_1)
                choice_2 = random.choice(vertices_2)
                if not bool_matrix[choice_1][choice_2]:
                    bool_matrix[choice_1][choice_2] = True
                    bool_matrix[choice_2][choice_1] = True
                    n_edges += 1
        super(MIS, self).__init__(n_vertices, n_edges, bool_matrix)

    def to_spn(self) -> SPN:
        """Transforms this MIS instance into an SPN"""
        neighbors = {}
        for i in range(self._n_vertices):
            neighbors[i] = []
            for possible_neighbor in range(self._n_vertices):
                if self._data[i][possible_neighbor]:
                    neighbors[i].append(possible_neighbor)

        product_nodes = []
        counts_for_c = np.zeros(self._n_vertices)
        number_of_neighbors = np.zeros(self._n_vertices)
        root_node = SumNode()

        for i in range(self._n_vertices):
            number_of_neighbors[i] = len(neighbors[i])
            counts_for_c[i] = 2 ** (self._n_vertices - number_of_neighbors[i] - 1)

            product_node = ProductNode()
            for j in range(self._n_vertices):
                prob = 1 if i == j else 0 if self._data[i][j] else 0.5
                child = MultinomialNode(Variable(j, 2), distribution=[1.0 - prob, prob])
                product_node.add_child(
                    child.to_sum_node()
                )
            product_nodes.append(product_node)

        norm_coeff = np.sum(counts_for_c)
        for i in range(self._n_vertices):
            weight = (2 ** (self._n_vertices - number_of_neighbors[i] - 1)) / norm_coeff
            root_node.add_child(product_nodes[i], weight)

        return root_node
