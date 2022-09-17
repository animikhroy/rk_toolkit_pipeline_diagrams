#!/usr/bin/env python3
import random

from ..visualizer import RKModelVisualizer, RKDiagram
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from rktoolkit.models.graph import Node, RKModel, GraphMask, GraphModel
import copy
import logging
import networkx as nx

'''
Cicular Visualizer Makes an almost "Mandlebot" like visualization
of the RKModel.

How it works: based on the center, and relative number of children nodes,
it expands outward using a circular pattern. As the expansion continues,
children birth more children in circular patterns.

The distance and sizes of of the node between each parent -> child in the visualization
decreases each iteration.

You can override any particular value using the attributes of a node.
'''

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
#    if not nx.is_tree(G):
#        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

class DendrogramVisualizerSpec():
    '''
    These are specs for the circular visualiation
    They define how close the clsuters are to the
    centroid, the size of the centroid bubble,
    alpha, etc.

    You can compute these dynamically or
    use static fields
    '''

    def __init__(self,
                 center_color='#000000',
                 center_size=300,
                 distance_from_center=10,
                 cluster_size=10,
                 add_node_labels = False,
                 alpha=.5):
        self.center_color = center_color
        self.center_size = center_size
        self.distance_from_center = distance_from_center
        self.alpha = alpha
        self.cluster_size = cluster_size
        self.add_node_labels = add_node_labels

class DendrogramVisualizer(RKModelVisualizer):
    '''
    Cicular Visualizer  visualizes an RKModel
    by plotting clusters and measures in a circular
    pattern in 3D space

    It assumes the RKModel is at least in 3d
    TODO: Visualizer only gives locations and specs.
    RK-Diagram Renderer renders the specs
    '''
    def __init__(self, spec: DendrogramVisualizerSpec = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.method = "dendrogram"

        if spec is None:
            spec = DendrogramVisualizerSpec()
        self.spec = spec
        self.positions = {}

    def _build(self, model: GraphModel):
        try:
            G = nx.DiGraph()
            [ G.add_node(n.id) for n in tr.nodes ] #TODO: A dendrogram
            for n in tr.nodes:
                for c in n.children:
                    G.add_edge( n.id, c.id)
            fig, ax = plt.subplots(figsize=(12,12))
            pos = hierarchy_pos(G,'root')
            nx.draw(G, pos=pos, with_labels=True, ax=ax)
        except Exception as e:
            print("Failed to create model")
            raise e
