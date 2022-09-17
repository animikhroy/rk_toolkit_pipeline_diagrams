from ..models.functions import LinkageFunction
from ..models.graph import HierarchicalGraph, Edge
import logging
import numpy as np
import itertools
from typing import Optional

class SimpleLinkageFunction(LinkageFunction):
    '''
    A greedy linkage function
    '''
    def __init__(self, threshold=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _recursive_link(self, parent, links = []):
        # make links between children
        # simple method: O(n^2)
        # better waays to do this and reduct to O(n(log(n)))
        if len(parent.children) > 1:
            for p in itertools.permutations(parent.children):
                p1, p2 = p[0], p[1]
                if p1.value is None or p2.value is None:
                    continue

                d = np.linalg.norm(p1.value - p2.value)
                if (self.threshold < 0 or d < self.threshold) and p2.value <= p1.value:
                    links.append(Edge(from_id=p2.id, to_id=p1.id,
                                weight=d, attributes={
                                    'delta': d,
                                }))

        # Make links between children -> parent
        for c in parent.children:
            if parent.is_root:
                 links.append(Edge(from_id=c.id, to_id=parent.id,
                             weight=1, attributes={}))
                 self._recursive_link(c, links)

            if parent.value is None:
                continue

            d = np.linalg.norm(parent.value - c.value)
            if (self.threshold < 0 or d < self.threshold) and c.value <= parent.value:
                links.append(Edge(from_id=c.id, to_id=parent.id,
                             weight=d, attributes={
                                 'delta': d,
                             }))

            return self._recursive_link(c, links)

        return links

    def link(self, X: HierarchicalGraph):
        return self._recursive_link(X.get_root(), [])
