#!/usr/bin/env python3
from pydantic import BaseModel, PrivateAttr
from enum import Enum
from typing import List, Optional, Callable
import uuid
from copy import deepcopy, copy

# Edge types for edges
class EdgeType(Enum):
    UNDIRECTED=1
    DIRECTED=2

class Edge(BaseModel):
    from_id: str
    to_id: str
    id: Optional[str]
    edge_type: EdgeType = EdgeType.UNDIRECTED
    weight: Optional[float]  = 1
    attributes: Optional[dict] = None
    '''
    A graph edge links nodes together
    using the nodeid.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())

class Node(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None,
    value: Optional[float] = None
    attributes: Optional[dict] = {}
    '''
    A node represents a distinct object in a graph

    A subclass of node that speiciflcally is used
    in DAG creation. It has a couple features a normal
    node doesn't have, such as checking boundary conditions
    and an executing function for data transformation
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())

class TreeNode(Node):
    parent: Optional[Node] = None
    children: Optional[List[Node]] = []
    is_root: Optional[bool] = False

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level


class TreeTransformNode(TreeNode):

    transformf: Optional[Callable] = lambda x: x
    fitf: Optional[Callable] = lambda X,y: None

    def __init__(self, **data):
        super().__init__(**data)

    def transform(self, X):
        return self.transformf(X)

    def fit(self, X, y):
        self.fit(X,y)

class PipelineNode(TreeNode):
    '''
    defines a pipeline node
    '''
    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

class GraphModel(BaseModel):
    '''
    The underlying graph model is a relationship
    of nodes and edges
    '''
    nodes: Optional[Node] = []
    edges: Optional[Edge] = []
    _nids: dict = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._nids = {}

    def add_node(self, node: Node):
        self.nodes.append(node)
        self._nids[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_node_by_id(self, id: str) -> Node:
        if id not in self._nids:
            raise ValueError("Node ID: {} not in graph.".format(id))
        return self._nids[id]

class HierarchicalGraph(GraphModel):
    '''
    A Hierarchical Graph
    is a subset of the general graph in which
    all elements are directed.
    '''
    root: TreeNode
    _level_ref: Optional[dict] = PrivateAttr()
    _node_ref: Optional[dict] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._level_ref = None
        self._node_ref = None
        self.root.is_root = True

    def add_edge(self, edge:Edge):
        if edge.edge_type != EdgeType.DIRECTED:
            raise ValueError("Edge type needs to be directed in heirarchial graph")

    def add_node(self, node: TreeNode):
        if not isinstance(node, TreeNode):
            raise ValueError("Node must be a tree node")
        if node.parent is not None:
            node.parent.children.append(node)
        super().add_node(node)

    def _build_levels(self) -> (dict(), dict()):
        '''
        Inefficient method to build the levels of a graph
        based. Returns two dictionaries:

        1: A dictionary where dict[level] -> [list of nodes]
        2. A dictionary where dict[id] -> level

        TODO: Build a more efficient data structure to index levels
        '''
        levels, nmap = {}, {}
        def _build_level_from_parent(n, d, d1, c):
            if c not in d:
                d[c] = []
            d[c].append(n)
            d1[n.id] = c
            for child in n.children:
                _build_level_from_parent(child, d, d1, c+1)
        _build_level_from_parent(self.root, levels, nmap, 0)
        return levels, nmap

    def get_root(self) -> Node:
        return self.root

    def get_level(self, level, rebuild=True) -> List[Node]:
        if self._level_ref is None or rebuild:
            self._level_ref, self._node_ref = self._build_levels()
        return self._level_ref.get(level, [])

class HierarchicalTransformGraph(HierarchicalGraph):
    '''
    Heirarchical Transform Graph contains
    transform functions
    '''
    def __init__(self, **data):
        super().__init__(**data)

    def _transform(self, parent, X):
        for n in parent.children:
            if isinstance(n, TreeTransformNode):
                v = n.transform(X)
                n.value = v
            self._transform(n, X)

    def transform(self, X):
        self._transform(self.get_root(), X)
        return self

class GraphMask(BaseModel):
    '''
    Graph masks are similar to numpy masks.
    They represent a masking of graph elements
    and are ultimately what the filter functions
    are generating.
    '''
    nodeMasks: dict = dict()
    edgeMasks: dict = dict()

    def node_is_masked(self, id):
        return self.nodeMasks.get(id, False)

    def edge_is_masked(self, id):
        return self.nodeMasks.get(id, False)

    def mask_node(self, node):
        self.nodeMasks[node.id] = True

    def unmask_node(self, node):
        self.nodeMasks[node.id] = False

    def mask_edge(self, edge):
        self.edgeMasks[edge.id] = True

    def unmask_edge(self, edge):
        self.edgeMasks[edge.id] = False

class RKModel(BaseModel):
    '''
    An RK-Model is the core data-structure that can be manifested in an
    RK-Diagram or interpreted.

    RK-Models are different from most models due to a number of important
    features:

    - HierarchicalGraph(h) => a heierarchical graph of features. Which features
    have a directional relationship with eachother
    - Mask (mask) =>  A graph mask. Represents a masking function on top of
    the Hierarchical Graph
    - Links => Links just edges. They can be computed with a Linkage Function
    - Locatoin => A global position for reference. When plotting an rkmodel against
    other rk-models, this positions the rk-models relative to eachtother
    '''
    mask: GraphMask = None
    hgraph: HierarchicalGraph = None
    links: List[Edge] = None
    location: List[float] = None
    name: Optional[str]

    def complete(self) -> bool:
        if self.mask is None or self.h is None \
           or self.links is None or self.location is None:
            return False
        return True
