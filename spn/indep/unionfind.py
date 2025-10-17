"""Union Find: An implementation of the union find algorithm"""

from typing import List, Optional


class UnionFindNode:
    """A node in an union-find algorithm"""

    def __init__(
        self,
        var_id: int,
        rank: int = 0,
        parent: "UnionFindNode" = None,
        children: List["UnionFindNode"] = None,
    ):
        if parent is None:
            parent = self
        if children is None:
            children = []
        self.__var_id = var_id
        self.__rank = rank
        self.__parent = parent
        self.__children = children

    @property
    def rank(self) -> int:
        """Rank is the measure for UnionFind"""
        return self.__rank

    @property
    def parent(self) -> "UnionFindNode":
        """Parent of this node. If parent is equal to the node, it doesn't
        have a parent"""
        return self.__parent

    @parent.setter
    def parent(self, new_parent: "UnionFindNode"):
        """Changes parent node of this node"""
        self.__parent = new_parent

    def add_child(self, child: "UnionFindNode"):
        """Adds a new child to this node"""
        self.__children.append(child)

    def increment_rank(self):
        """Ups node's rank by 1"""
        self.__rank += 1

    def find(self) -> "UnionFindNode":
        """Returns the root (parent of parents) for this node.
        Also assigns the root of this node's parent as its root"""
        if self is not self.__parent:
            self.__parent = self.__parent.find()
        return self.__parent

    def var_ids(self) -> List[int]:
        """Returns a list containing the id of the variable in this node and the
        ids of the variables in its children"""
        var_ids = [self.__var_id]
        for child in self.__children:
            var_ids.extend(child.var_ids())
        return var_ids


def node_union(node1: UnionFindNode, node2: UnionFindNode) -> Optional[UnionFindNode]:
    """Returns a node representing the union of nodes 1 and 2"""
    node1 = node1.find()
    node2 = node2.find()
    if node1 is node2:
        return None
    if node1.rank > node2.rank:
        node2.parent = node1
        node1.add_child(node2)
        return node1
    node1.parent = node2
    node2.add_child(node1)
    if node1.rank == node2.rank:
        node2.increment_rank()
    return node2
