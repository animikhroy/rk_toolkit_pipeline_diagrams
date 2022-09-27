#!/usr/bin/env python
import uuid
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d

legends = {} #HACK

class EventScape:

    def __init__(self, events=None):
        events = [] if events is None else events
        self.events = events
        self._id = uuid.uuid4()

    def add_event(self, event):
        self.events.append(event)

    def merge_clusters(self):
        pass

    def nclusters(self):
        return len(self.events)

    def events():
        return self.events

    def visualize(self, ax, x=True, y=True, z=True, random=False, **kwargs):
        event_colors = {
            'GW170817': 'blue',
            'GW190814': 'magenta',
            'GW190521': 'cyan',
            'GW170729': 'brown'
        }

        for i, e in enumerate(self.events):
            # get GPS time
            center=[e.attributes['ts'], e.attributes['pf'],e.attributes['snr']]
            if not z:
                center[2] = 0
            if not y:
                center[1] = 0
            if not x:
                center[0] = 0
            e.visualize(ax, center=center, x=x,y=y,z=z, show_legend=(i==0),
                        center_color=event_colors[e.name], random=random, **kwargs)

        handlers = []
        for n, c in event_colors.items():
            red_patch = mpatches.Patch(color=c, label=n)
            handlers.append(red_patch)
        for l, c in legends.items():
            handlers.append(mpatches.Patch(color=c, label=l))

        ax.legend(handles=handlers)

class RKEvent:

    def __init__(self, name="", clusters=None, attributes=None):
        clusters = [] if clusters is None else clusters
        self.clusters = clusters
        self._id = uuid.uuid4()
        if attributes is None:
            attributes = {}
        self.attributes = attributes
        self.name = name

    def add_cluster(self, cluster):
        self.clusters.append(cluster)

    def merge_clusters(self):
        pass

    def visualize(self, ax, x=True, y=True, z=True, show_legend=True,
                  center_color='black', random=False, **kwargs):
        colors = ['red', 'pink', 'green', 'yellow', 'orange','purple', 'yellow', 'maroon']
        count = 0
        for i, cluster in enumerate(self.clusters):
            cluster.visualize(ax, color=colors[i], center_color = center_color,
                              random=random,
                              count_offset = count, n_clusters=len(self.clusters)+2,
                            angle_width=2*np.pi/(len(self.clusters)),
                              angle_offset=i*2*np.pi/(len(self.clusters)), x=x, y=y, z=z, show_legend = show_legend, **kwargs)
            count += len(cluster.get_nodes())


class RKCluster(nx.Graph):

    def __init__(self, name="", nodes=None, edges=None):
        self._nodes = nodes if nodes is not None else []
        self._edges = edges if edges is not None else []
        self._g = nx.Graph()
        self._name = name
        self._id = uuid.uuid4()
        self.visualized_index = 0

    def get_node_by_id(self, id):
        for n in self.get_nodes():
            if id == n.get_id():
                return n
        return None

    def compute_centroid(self):
        pass

    def add_graph(self, graph):
        self._g = graph

    def add_node(self, node=None):
        self._nodes.append(node)

    def add_edge(self, edge):
        self._edges.append(edge)

    def ecount(self):
        return len(self._edges)

    def ncount(self):
        return len(self._nodes)

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def visualize(self, ax, distance_from_center=10, center_size=300,
                  random=False,
                  center=[0,0,0], count_offset=0, n_clusters=10,
                  center_color='black', angle_width=2*np.pi,
                  angle_offset=0, in3d=True, color='blue', x=True, y=True, z=True, show_legend=True,
                  show_minor=False,
                  **kwargs):
        '''
        Viualizes the RK Cluster. Recommended to make axis 3d.

        :params ax: matplotlib ax
        :params distance_from_center: distance to draw nodes from center
        :params center_size: the size of the center node
        :params center: Location to center cluster
        :params in3d: to draw in 3d
        :color: color of the cluster.
        '''

        # compute cluster center
        # draw center
        #
        if show_minor:
           # fig2, minor_ax = plt.subplots()
            fig = plt.figure(dpi=100)

            minor_ax = Axes3D(fig)

            plt.grid(b=None)
            #plt.axis('off')
            # make the panes transparent
            # minor_ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # minor_ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # minor_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            # minor_ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # minor_ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # minor_ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

            fig = plt.figure(dpi=100)

            minor_ax2 = Axes3D(fig)

            plt.grid(b=None)
            #plt.axis('off')
            # make the panes transparent
            # minor_ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # minor_ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # minor_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            # minor_ax2.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # minor_ax2.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # minor_ax2.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        if not x:
            center[0] = 0
        if not y:
            center[1] = 0
        if not z:
            center[2] = 0

        if in3d:
            ax.scatter(xs=center[0], ys=center[1], zs=center[2], c=center_color, s=center_size, alpha=.5)
        if show_minor:
            minor_ax.scatter(xs=center[0], ys=center[1], zs=center[2], c=center_color, s=center_size, alpha=.5)
#            minor_ax2.scatter(xs=center[0], ys=center[1], zs=center[2], c=center_color, s=center_size, alpha=.5)


        rads = angle_width/(self.ncount() + 1)

        for i, node in enumerate(self.get_nodes()):
            # compute the location from the angle
            #                 cangle = angle_offset + (rads * i) + (np.random.rand() * rads)
            cangle2 = angle_offset + (rads * i) + (np.random.rand() * rads)

            if random:
                cangle = angle_offset + (rads * i) + (np.random.rand() * rads)
            else:
                cangle = (rads * i) + angle_offset

            # hardcoded to normalize
            xdist = distance_from_center * 1000000
            ydist = distance_from_center * 80
            zdist = distance_from_center * 30

            if not random:
                if x and y and z:
                    loc_xyz = np.array(center) + [0, np.cos(cangle)*ydist,
                                                np.sin(cangle)*zdist]
                elif not x and y and z:
                    loc_xyz = np.array(center) + [0,np.cos(cangle)*ydist,
                                                np.sin(cangle)*zdist]
                elif x and not y and z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist, 0,
                                                np.sin(cangle)*ydist]
                elif x and y and not z:
                    loc_xyz = np.array(center) + [np.cos(cangle) * xdist,
                                                np.sin(cangle)*ydist, 0]
                elif not x and y and not z:
                    loc_xyz = np.array(center) + [0, np.cos(cangle)*ydist,
                                                np.sin(cangle)*zdist]
                elif x and not y and not z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist,
                                                np.sin(cangle)*ydist, 0]
                elif not x and not y and z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist, 0,
                                                np.sin(cangle)*zdist]
            else:
                if x and y and z:
                    loc_xyz = np.array(center) + [np.random.rand() * xdist, np.cos(cangle)*ydist,
                                                np.sin(cangle2)*zdist]
                elif not x and y and z:
                    loc_xyz = np.array(center) + [np.random.rand() * xdist, np.cos(cangle)*ydist,
                                                np.sin(cangle)*zdist]
                elif x and not y and z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist, 0,
                                                np.sin(cangle)*ydist]
                elif x and y and not z:
                    loc_xyz = np.array(center) + [np.cos(cangle) * xdist,
                                                np.sin(cangle)*ydist, 0]
                elif not x and y and not z:
                    loc_xyz = np.array(center) + [0, np.cos(cangle)*ydist,
                                                np.sin(cangle)*zdist]
                elif x and not y and not z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist,
                                                np.sin(cangle)*ydist, 0]
                elif not x and not y and z:
                    loc_xyz = np.array(center) + [np.cos(cangle)*xdist, 0,
                                                np.sin(cangle)*zdist]

            node.attributes['pos'] = loc_xyz

            if in3d:
                xx = np.array((loc_xyz[0], center[0]))
                yy = np.array((loc_xyz[1], center[1]))
                zz = np.array((loc_xyz[2], center[2]))
                ax.plot(xx, yy, zz, c='black', alpha=0.5)
                if show_minor:
                    minor_ax.plot(xx, yy, zz, c='black', alpha=0.5)

            if in3d:
                label = node.attributes['name_of_cluster']
                if label in legends or not show_legend:
                    label = ""
                ax.scatter(xs=loc_xyz[0], ys=loc_xyz[1], zs=loc_xyz[2],
                           c=color, s=100, edgecolors='k', alpha=0.7, zorder=2, label=label)
                if label != "":
                    legends[label] = color

                if show_minor:
                    minor_ax.scatter(xs=loc_xyz[0], ys=loc_xyz[1], zs=loc_xyz[2],
                           c=color, s=100, edgecolors='k', alpha=0.7, zorder=2, label=label)
                    minor_ax2.scatter(xs=loc_xyz[0], ys=loc_xyz[1], zs=loc_xyz[2],
                           c=color, s=100, edgecolors='k', alpha=0.7, zorder=2, label=label)

        print("Drawing {} edges for {} nodes".format(len(self.get_edges()), len(self.get_nodes())))

        drawn_ids = set()
        for i, edge in enumerate(self.get_edges()):
            # plot the edges
            fromnode = self.get_node_by_id(edge.from_id)
            tonode = self.get_node_by_id(edge.to_id)

            eid = "{}".format(sorted('_'.join([str(edge.from_id), str(edge.to_id)])))
            if eid in drawn_ids:
                print("Already drawn")
            if fromnode is None or tonode is None:
                raise ValueError("Could not find Node!")

            pos_1 = fromnode.attributes['pos']
            pos_2 = tonode.attributes['pos']

            x = np.array((pos_1[0], pos_2[0]))
            y = np.array((pos_1[1], pos_2[1]))
            z = np.array((pos_1[2], pos_2[2]))

            if in3d:
                ax.plot(x, y, z, c='black', alpha=0.5)
                drawn_ids.add("{}".format(eid))
                a = Arrow3D([pos_2[0], pos_1[0]], [pos_2[1], pos_1[1]],
                    [pos_2[2], pos_1[2]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color='black', zorder=3)
                ax.add_artist(a)

                if show_minor:
                    a = Arrow3D([pos_2[0], pos_1[0]], [pos_2[1], pos_1[1]],
                                [pos_2[2], pos_1[2]], mutation_scale=10,
                                lw=1, arrowstyle="-|>", color='black', zorder=3)
                    minor_ax.add_artist(a)


        # lines to center
        for i, node in enumerate(self.get_nodes()):
            x = np.array((node.attributes['pos'][0], center[0]))
            y = np.array((node.attributes['pos'][1], center[1]))
            z = np.array((node.attributes['pos'][2], center[2]))
            if in3d:
                ax.plot(x, y, z, c='black', alpha=0.5)

            if show_minor:
                ax.plot(x, y, z, c='black', alpha=0.5)

        self.visualized_index += 1

        print("Drew {} edges for {} nodes".format(len(drawn_ids), len(self.get_nodes())))
        return ax


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
