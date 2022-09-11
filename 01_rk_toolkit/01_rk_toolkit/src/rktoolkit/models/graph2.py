# Maintainer: andor@henosisknot.com
# Main graph class

from networkx.classes.digraph import DiGraph
import networkx as nx

def jaccard(s1, s2): # two sets
    intersect = s1 & s2
    jin = len(intersect)  / (len(s1) + len(s2) - len(intersect))
    return 1-jin

class GraphMask():

    def __init__(self, nmasks=[], emasks=[]):
        self._nmask = set(nmasks)
        self._emask = set(emasks)

    def get_nmasks(self, n):
        return self._nmask

    def get_emasks(self, n):
        return self._emask

    def fit(self, G):
        sgraph = self._nmask ^ set(list(g.nodes.keys()))
        mG = G.__class__()
        mG.add_nodes_from((n, G.nodes[n]) for n in sgraph)
        mG.add_edges_from(G.edges)
        edges = dict(mG.edges.items())
        for e in self._emask:
            if e in edges:
                mG.remove_edge(*e)
        return mG

class Graph(DiGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

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

    def get_children(self, node_id):
        return nx.descendants(g, node_id)

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
            s1 = set(list(self.nodes.keys()))
            s2 = set(list(G.nodes.keys()))
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
        return (vd + td) / 2

    def value_distance(self, G, method="cossine", key="value"):
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

    def __init__(self, id: str, value=None):
        self.value = value
        self.id = id

    def to_dict(self):
        return {
            "id": self.id,
            "value": self.value
        }
