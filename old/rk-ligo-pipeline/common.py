from networkx.classes.digraph import DiGraph
import networkx as nx 
import numpy as np

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


# +
import copy
import numbers 

class RKModel():
    
    def __init__(self, G, mask, edges):
        self.G = G
        self.mask = mask
        self.edges = edges
        
    def get(self):  
        gC = copy.deepcopy(self.G)
        for k,v in self.edges.items():
            gC.add_edge(Edge(*k, attributes=v))
        return GraphMask(nmasks=self.mask).fit(gC)
    
class RKPipeline():
    
    def check_valid_node(self, node) -> bool:
        if "value" not in node:
            return False   
        if not isinstance(node["value"], numbers.Number):
            return False
        return True

    def __init__(self, filter_map: dict, linkage_map: dict, structural_graph = None):
        self.filter_map = filter_map
        self.linkage_map = linkage_map
        self.structural_graph = structural_graph

    def transform(self, G, is_base=True):
        if is_base:
            self.structural_graph = G
        gC = copy.deepcopy(G)
        for k, v in self.linkage_map.items():
            gC = v.link(G)
        masks = set()
        for n, f in self.filter_map.items():
            if self.check_valid_node(self.structural_graph.nodes[n]):
                if f.filter(gC.nodes[n]):
                    for child in G.get_children(n, recursive=True):
                        masks.add(child)
                    masks.add(n)
                
        return RKModel(self.structural_graph, list(masks), gC.edges)
    
    def remap(self, vmap, cols):
        pcopy = copy.deepcopy(self)
        for i, v in enumerate(vmap):
            col = cols[i]
            knb = col.split("_", maxsplit=2)[1]
            key = col.split("_", maxsplit=2)[2]
            if col.split("_")[0] == "filter":
                pcopy.filter_map[key].set_knob(knb, v)
            if col.split("_")[0] == "linkage":
                pcopy.linkage_map[key].set_knob(knb, v)
        return pcopy
    
    def get_w(self):
        vmap, cols = [], []
        for k,v in self.filter_map.items():
            knbs = v.get_knobs()
            for i, l in knbs.items():
                cols.append('{}_{}_{}'.format('filter', i, k))
                vmap.append(l)
        for k,v in self.linkage_map.items():
            knbs = v.get_knobs()
            for i, l in knbs.items():
                cols.append('{}_{}_{}'.format('linkage', i, k))
                vmap.append(l)
        return vmap, cols
# -


