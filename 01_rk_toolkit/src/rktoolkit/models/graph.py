# Maintainer: andor@henosisknot.com
# Main graph class
from networkx.classes.digraph import DiGraph
import networkx as nx
import copy
import numbers
from ..functions.distance import jaccard, mahalanobis
from pydantic import BaseModel, PrivateAttr
from enum import Enum
from typing import List, Optional, Callable
import uuid
from copy import deepcopy, copy

class GraphMask():

    def __init__(self, nmasks=[], emasks=[]):
        self._nmask = set(nmasks)
        self._emask = set(emasks)

    def get_nmasks(self, n):
        return self._nmask

    def get_emasks(self, n):
        return self._emask

    def fit(self, G):
        sgraph = self._nmask ^ set(list(G.nodes.keys()))
        mG = G.__class__()
        mG.add_nodes_from((n, G.nodes[n]) for n in sgraph)
        for e in list(G.edges):
            if e[0] in sgraph and e[1] in sgraph:
                mG.add_edges_from([e])

        edges = dict(mG.edges.items())
        for e in self._emask:
            if e in edges:
                mG.remove_edge(*e)
        return mG

class Graph(DiGraph):

    def __init__(self, id=None, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.id = id

    def add_vertex(self, n):
        if isinstance(n, Vertex):
            super().add_nodes_from([(n.id, n.to_dict())])
        else:
            raise ValueError("Expected a Vertex Type")

    def add_edge(self, e):
        if isinstance(e, Edge):
            super().add_edges_from([e.to_dict()])
        else:
            raise ValueError("Expected a Edge Type")

    def get_children(self, node_id, recursive=False):
        if recursive:
            return nx.descendants(self, node_id)
        else:
            return list(self.successors(node_id))

    def validate(self):
        try:
            if not self.is_connected():
                raise ValueError("Graph is not connected")
            if not self.is_dag():
                raise ValueError("Not a DAG")
            self._find_cycles()
        except Exception as e:
            raise e
        return True

    def is_connected(self):
        return nx.is_connected(self.to_undirected())

    def edge_distance(self, G, method="jaccard"):
        if method == "jaccard":
            s1 = set(list(self.edges.keys()))
            s2 = set(list(G.edges.keys()))
            return jaccard(s1, s2)
        raise ValueError("Unknown method to compute distances")

    def node_distance(self, G, method="jaccard"):
        if method == "jaccard":
            s1 = set(list(self.nodes.keys()))
            s2 = set(list(G.nodes.keys()))
            return jaccard(s1, s2)
        raise ValueError("Unknown method to compute distances")

    def topological_distance(self, G, method="jaccard", weights=[.5,.5]):
        ed = self.edge_distance(G, method)
        nd = self.node_distance(G, method)
        return sum([ed * weights[0], nd * weights[1]]) / 2

    def weighted_distance(self, G, topological_method="jaccard", value_method="cossine", key="value", weights=[.5, .5]):
        td = self.topological_distance(G, method=topological_method, weights=[.5, .5])
        vd = self.value_distance(G, method=value_method, key="value")
        logger.info("Got value distance {} and toplogical distance {}".format(td, vd))
        return ((vd + td) / 2)[0]

    def similarity(self, *args, **kwargs): #returns the inverse of the normalized distance.
        return 1- self.weighted_distance(*args, **kwargs)

    def value_distance(self, G, method="cossine", key="value", fillValue=0):
        a1 = self.get_value_dict(key=key)
        a2 = G.get_value_dict(key=key)
        isect = a1.keys() & a2.keys()
        # build array
        arr1 = []
        arr2 = []
        for i in isect:
            arr1.append(a1[i])
            arr2.append(a2[i])
        arr1 = np.array(arr1).reshape(1, -1)
        arr2 = np.array(arr2).reshape(1, -1)
        arr1 = arr1.astype(float)
        arr1[np.isnan(arr1)] = fillValue
        arr2 = arr2.astype(float)
        arr2[np.isnan(arr2)] = fillValue
        return 1 - cosine_similarity(arr1, arr2)

    def is_dag(self):
        return nx.is_directed_acyclic_graph(self)

    def sort(self, *args, **kwargs):
        return nx.topological_sort(self, *args, **kwargs)

    def get_value_dict(self, key="value"):
        return { n[0]: n[1].get(key, 0) for n in self.nodes.items()}

    def _find_cycles(self):
        try:
            nx.find_cycle(self, orientation=None)
        except nx.exception.NetworkXNoCycle:
            return True
        except Exception as e:
            raise e
        raise Exception("Cycle Detected! Invalid Graph.")

class Edge():

    def __init__(self, u, v, attributes={}):
        self.u = u
        self.v = v
        self.attributes = attributes

    def to_dict(self):
        return (self.u, self.v, self.attributes)

class Vertex():

    def __init__(self, id: str, value=None, attributes={}):
        self.value = value
        self.id = id
        self.attributes = attributes

    def to_dict(self):
        return {
            "id": self.id,
            "value": self.value,
            **self.attributes
        }

'''
Older Stuff
TODO: figure out how much is still used and remove bad components
'''
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

