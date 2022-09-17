from ..models.graph import HierarchicalTransformGraph, TreeNode, TreeTransformNode
import pandas as pd


class BaseOntologyTransform():
    '''
    Given an ontology, creates the transform
    '''

    def __init__(self, mapping, lens="root", color_decay_rate=.1):
        self.mapping = mapping
        self.lens = lens
        self.cmap = matplotlib.cm.get_cmap('Spectral')
        self.color_decay_rate = color_decay_rate

    def transform(self, X):
        H = Graph()
        H.add_vertex(Vertex(self.lens))
        return self._convert(X, H, self.mapping, parent=self.lens, level=1)

    def _convert(self, X, H, hmap=None, parent=None, level=0, color=None, lens="root"):
        count = 0
        for k, v in hmap.items():
            if level == 1:
                color = np.array(self.cmap(count/len(hmap.keys())))
            value = X[k] if k in X else None
            color[3] *= 1- self.color_decay_rate
            node = Vertex(id=k, attributes={"color": color}, value=value)
            H.add_vertex(node)
            H.add_edge(Edge(u=parent, v=k))
            self._convert(X, H, v, parent=k, level=level+1, color=color)
            count+=1
        return H

class CorrelationHTGGenerator():

    def __init__(self, threshold=.7):
        self.threshold = threshold
        self._corr = None

    def fit(self, X,y):
        pass

    def transform(self, X):

        cdf = pd.DataFrame(X).corr().abs()
        measures = TreeNode(parent=None, id='measures')
        H = HierarchicalTransformGraph(root=measures)

        nodes = []
        for i,v in cdf.iterrows():

            parent = TreeNode(name="{}_measure".format(i), id="{}_measure".format(i),
                    parent=measures)
            j = i

            nodes.append(parent)

            n_id = "{}_{}".format(i,i)
            n1 = TreeTransformNode(name=n_id, id=n_id, parent=parent,
                                   transformf=lambda X: X[i])
            nodes.append(n1)

            while j < len(v) - 1:

                if j == i:
                    j+=1
                    continue

                if v[j] > self.threshold:
                    n_id = "{}_{}".format(i,j)
                    n1 = TreeTransformNode(name=n_id, id=n_id, parent=parent,
                                           transformf=lambda X: X[j])
                    nodes.append(n1)
                j+=1
        [H.add_node(n) for n in nodes]
        return H
