from typing import List
from .graph import Edge
from .graph import Node

class LinkageSpec():
    pass

class LinkageFunction():
    '''
    A linkage function takes in a list of nodes
    and returns a list of edges

    There are mnay different types of linkage functions.

    For example, see https://pypi.org/project/fastcluster/
    as an example

    We have in the LIGO example, a linkage function with
    Euclidean distance defined.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'link') and callable(subclass.link)

    def link(self, nodes: List[Node]) -> List[Edge]: # given a graph the edges
        return []
