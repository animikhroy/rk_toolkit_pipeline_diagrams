from rk_diagram.models import RKPipeline, LocalizationAlgorithm, TransformNode
from rk_diagram.visualize import RKModelVisualizer
from rk_diagram.models.graph import EdgeType, Edge

import numpy as np

class HierarchicalFeatureExtractor1():
    '''
    Generates a heirarchical feature extractor
    TODO: Think about 2+ levels
    '''
    def __init__(self):
        self.children = {}

    def predict(self, X):
        self.children['range_measure'] = range_measure(X)
        self.children['max_measure'] = max_measure(X)

    def range_measure(self, X): # computes the range as a feature
        return np.max(X) - np.min(X)

    def max_measure(self, X): # computes the max as a measure
        return np.max(X)


class SimpleLinkage():
    '''
    Simple linkage:
    A simple linkage function.
    Compares the values of two nodes, draws a link if the euclidean distancd is
    less than the threshold.

    Sends back a list of edges
    '''
    def __init__(self, threshold):
        self.threshold = 5

    def link(self, nodes):
        edges = []
        for n, i in enumerate(nodes):
            l = n+1
            while l < len(nodes):
                if np.linalg.norm(nodes[i].value - nodes[l].values) < self.threshold:
                    l_larger = nodes[i].value > nodes[l].value
                    fid = nodes[l].from_id if l_larger else nodes[i].from_id
                    tid = nodes[i].from_id if l_larger else nodes[j].from_id
                    etype = EdgeType.DIRECTED
                    if nodes[i].value == nodes[l].value:
                        etype = EdgeType.UNDIRECTED
                    edges.append(Edge(from_id=fid, to_id=tid, type=etype))
                l+=1

class MaxLocalizer(LocalizationAlgorithm):
    '''
    localizes the max position
    '''
    def localize(self, X):
        return np.argmax(X) # returns the max position of X

class MinMaxNormalizerNode(TransformNode):
    '''
    min max normalizer
    takes the max and min as a transform node
    and will normalize the data
    '''
    def __init__(self):
        self._fitted = False

    def fit(self, X):
        self._fitted = True
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

class StaticFilter():
    '''
    This static filter takes simple boundary conditions,
    a min and max, and provides a filter function over it
    '''
    def __init__(self, min=None, max=None):
        self._min = min
        self._max = max

    def filter(self, val):
        if self._min is not None and val < self._min:
            return False
        if self._max is not None and val > self._max:
            return False
        return True

def main(X):

    rk_models = []
    example_pipeline = RKPipeline(preprocess_nodes=[MinMaxNormalizerNode()],
                                  localization_algorithm=MaxLocalizer(),
                                  hierarchical_embedding_nodes= [
                                      {
                                          "HFeatureExtractor1": HierarchicalFeatureExtractor1()
                                      }
                                  ],
                                  filter_functions=[
                                      {
                                          "HFeatureExtractor1" :
                                          {
                                              'range_measure': StaticFilter(min=.2, max=.8),
                                              'max_measure': StaticFilter(min=0, max=1)
                                          }
                                       }
                                  ], # question: how to define which limits for which measure. Each filter and linkage has to be BY CLUSTER
                                  linkage_function=SimpleLinkage(threshold=.8))
    example_pipeline.build()
    example_pipeline.fit(X)
    rk_model = example_pipeline.transform(X)
    rk_models.append(rk_model)

    visualizer = RKModelVisualizer(method="circular")
    visualizer.build(rk_models) # build requires a list of rk_models
    visualizer.show()

def parse_arguments():
    X = [1,2,3,4]
    main()
