"Base: Contains the interface for generic Graphs"


class Base:
    """Class to represent a graph"""

    def __init__(self, n_vertices, n_edges, data):
        self._n_vertices = n_vertices
        self._n_edges = n_edges
        self._data = data

    @property
    def n_vertices(self):
        """Returns the number of vertices from this graph"""
        return self._n_vertices

    @property
    def n_edges(self):
        """Returns the number of edges from this graph"""
        return self._n_edges

    @property
    def data(self):
        """Returns a bool matrix where every position (i, j) symbolizes the
        presence of an edge between vertices i and j"""
        return self._data
