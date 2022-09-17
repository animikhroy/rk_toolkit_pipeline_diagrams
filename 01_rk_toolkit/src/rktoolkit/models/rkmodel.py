import copy
import numbers
from .graph import Edge, GraphMask

class RKModel():
    '''
    RK model
    contains the structural graph
    the mask
    and the edges
    '''
    def __init__(self, G, mask, edges):
        self.G = G
        self.mask = mask
        self.edges = edges

    def get(self):
        gC = copy.deepcopy(self.G)
        for k,v in self.edges.items():
            gC.add_edge(Edge(*k, attributes=v))
        return GraphMask(nmasks=self.mask).fit(gC)
