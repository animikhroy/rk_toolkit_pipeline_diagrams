from .graph import HierarchicalGraph, Edge, Node
from typing import List

'''
Function abstractions
'''
class LocalizationFunction():

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'localize') and callable(subclass.localize))

    def localize(self, X) -> [float]:
        '''
        Predicts the localized position
        Returns a list of float
        '''
        return

    def fit(self):
        '''
        Fits the data
        '''
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
        return (hasattr(subclass, 'predict') and callable(subclass.link))

    def predict(self, nodes: List[Node]) -> List[Edge]: # given a graph the edges
        return []

class FilterFunction():

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'filter') and callable(subclass.filter))

    def filter(self, X) -> bool:
        '''
        Predicts the localized position
        Returns a list of float
        '''
        return

    def fit(self):
        '''
        Fits the data
        '''
        pass
