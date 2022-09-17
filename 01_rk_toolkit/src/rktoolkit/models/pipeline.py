from pydantic import BaseModel
from .graph import (
    HierarchicalTransformGraph,
    TreeTransformNode,
    GraphMask,
)

from .rkmodel import RKModel
from .functions import *
from typing import List, Optional, Callable, TypedDict
from ..functions.localizers import IterableLocalizationFunction
from ..functions.linkers import SimpleLinkageFunction
from ..functions.htg_transformers import CorrelationHTGGenerator
from ..functions.filters import RangeFilter

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
