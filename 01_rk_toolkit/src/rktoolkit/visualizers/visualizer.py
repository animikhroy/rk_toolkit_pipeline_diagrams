from ..models.rkmodel import RKModel
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import uuid
from copy import copy, deepcopy

class RKModelVisualizer():
    '''
    Base classe for RK models
    Visualizes a model as an RK-Diagram
    '''
    def __init__(self, ax=None, fig=None):

        if fig is None:
            fig = plt.figure(dpi=100)

        if ax is None:
            ax = Axes3D(fig)

        if not isinstance(ax, Axes3D):
            raise ValueError("RKModel renders in 3d. Please Make sure to specify a 3d subplot")

        self.fig = fig
        self.ax = ax
        self.id = str(uuid.uuid4())
        self.method = "unspecified"

    def build(self, models: RKModel):
        self._build(models[0])

    def render(self):
        plt.show(block=True)


class Arrow3D(FancyArrowPatch):
    '''
    Builds arrows in 3d
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class RKDiagram():
    '''
    An RKDiagram is the manifestation of an RK-Model

    It contains the positions of each node, information about how
    to represent such nodes, as well as edge visualizations.

    It is purely a visualization diagram. To work effectively, the visualized
    space must be in nD < 4
    '''
    def __init__(self, rkmodel: RKModel, placed_nodes, links):
        self.rkmodel = rkmodel
        self.placed_nodes = placed_nodes
        self.links = links

    def render(self):
        # render specs here are done
        plt.show()
