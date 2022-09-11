from ..models.graph import HierarchicalTransformGraph, TreeNode, TreeTransformNode
import pandas as pd

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
