#!/usr/bin/env python
import tkinter as tk
import logging
import h5py
import json
import math
import readligo as rl
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from gwosc.locate import get_urls
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from rkmodel import filters
from rkmodel.spectrogram import Spectrogram
from gwpy.timeseries import TimeSeries
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rkmodel.graph import Node, Edge
from rkmodel.rkmodels import EventScape, RKEvent, RKCluster, Arrow3D
import pandas as pd
import requests
import os
from tkinter import messagebox

logger = logging.getLogger()
logger.setLevel(logging.INFO)

event_names = {'GW170729': {'path':'./H-H1_LOSC_4_V1-1167559920-32.hdf5', 'ts': 1185389807.3},
               'GW170817': {'path':'./H-H1_LOSC_4_V1-1167559920-32.hdf5', 'ts': 1187008882.43},
               'GW190521': {'path':'./H-H1_LOSC_4_V1-1167559920-32.hdf5', 'ts': 1242442967.4},
               'GW190814': {'path':'./H-H1_LOSC_4_V1-1167559920-32.hdf5', 'ts': 1249852257.0}}

root = tk.Tk()



class RKModelApplication(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side = tk.RIGHT)
        self.visualizer = RKGraphVisualizer(master=self)
        self.toolbar = Toolbar(master=self)
        self.toolbar.pack(side=tk.RIGHT)
        self.visualizer.pack(side=tk.LEFT)
        self.pack()

class RKGraphVisualizer(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.plot_line()
        self.figure = None

    def plot_line(self):
        pass
#        self.figure = plt.Figure(figsize=(6,5), dpi=100)
#        ax = self.figure.add_subplot(111)
#        chart_type = FigureCanvasTkAgg(self.figure, self.master)
#        chart_type.get_tk_widget().pack(side=tk.RIGHT)

    def get_strain(self, fn, t0):
        strain = TimeSeries.read(fn,format='hdf5.losc')
        center = int(t0)
        strain = strain.crop(center-16, center+16)
        return strain

    def plot_strain(self):
        figure = plt.gcf()
        figure.clf()
        figure.add_subplot(221)
        figure.add_subplot(222)
        figure.add_subplot(223)
        figure.add_subplot(224)
        axes = figure.get_axes()
        count=0
        figure.suptitle("Strain Data")
        for k,f in event_names.items():
            row = math.floor(count/2)
            col = count%2
            fn = f['file']
            if fn == "":
                return
            strain = self.get_strain(fn, f['ts'])
            axes[count].plot(strain)
            count+=1
        plt.show()

    def plot_asd(self):
        figure = plt.gcf()
        figure.clf()
        figure.add_subplot(221)
        figure.add_subplot(222)
        figure.add_subplot(223)
        figure.add_subplot(224)
        axes = figure.get_axes()
        count=0
        figure.suptitle("ASD Data")
        for k,f in event_names.items():
            ax = axes[count]

            fn = f['file']
            if fn == "":
                return

            strain = self.get_strain(fn, f['ts'])
            f2 = strain.asd(fftlength=8)
            ax.plot(f2)
            ax.set_xlim(10,2000)
            ax.set_ylim(1e-24, 1e-19)
            ax.set_yscale('log')
            ax.set_xscale('log')
            count+=1
        plt.show()

    def plot_eventscape(self):
        count=0

        xticksl = []
        dt=3
        full_eventscape = []

        for k,f in event_names.items():
            fn = f['file']
            strain = self.get_strain(fn, f['ts'])
            dt = 1
            t0 = f['ts']
            hq = strain.q_transform(outseg=(t0-dt, t0+dt))
            xticksl.extend(hq.xindex.to_value())
            if len(full_eventscape) == 0:
                full_eventscape = np.array(hq)
            else:
                full_eventscape = np.concatenate([full_eventscape, np.array(hq)])

        xticks = np.linspace(0,2.4, 4000)
        yticks = np.linspace(0,1000, 2560)
        fig, ax = plt.subplots()
        ax.pcolormesh(xticks, yticks, full_eventscape.T)
        ax.set_xticklabels(xticksl)
        ax.set_title("Merged Events")

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        H = full_eventscape.T
        cmap='viridis'
        X, Y = np.meshgrid(xticks, yticks)
        surf = ax.plot_surface(X, Y, H+10, cmap='viridis')
        cset = ax.contourf(X, Y, H, zdir='z', offset=np.min(H)-2, cmap=cmap)

        bar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        bar.set_label("Spectral Density")
        ax.set_xlabel('GPS Times')
        ax.set_zlabel('Merger Frequencies')
        ax.set_title('PBH Event Mergers: Shadow Network + Topological Landscape')
        plt.grid(b=None)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.set_xticklabels(xticksl)

        plt.xticks(rotation=-45)
        ax.view_init(30, 80)
        plt.tight_layout()
        plt.show()

    def plot_filteredscape(self):
        count=0

        xticksl = []
        dt=3
        full_eventscape = []

        for k,f in event_names.items():
            fn = f['file']
            strain = self.get_strain(fn, f['ts'])
            dt = 1
            t0 = f['ts']
            hq = strain.q_transform(outseg=(t0-dt, t0+dt))
            hq[hq < hq.max()] = 0
            xticksl.extend(hq.xindex.to_value())
            if len(full_eventscape) == 0:
                full_eventscape = np.array(hq)
            else:
                full_eventscape = np.concatenate([full_eventscape, np.array(hq)])

        from scipy.ndimage import gaussian_filter, maximum_filter
        H = full_eventscape.T
        H = maximum_filter(H, size=50)
        H = gaussian_filter(H, sigma=50)

        xticks = np.linspace(0,2.4, 4000)
        yticks = np.linspace(0,1000, 2560)
        fig, ax = plt.subplots()
        ax.pcolormesh(xticks, yticks, H)
        ax.set_xticklabels(xticksl)
        ax.set_title("Filtered Merged Events")
        #plt.show()

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        cmap='viridis'
        X, Y = np.meshgrid(xticks, yticks)
        surf = ax.plot_surface(X, Y, H+10, cmap='viridis')
        cset = ax.contourf(X, Y, H, zdir='z', offset=np.min(H)-2, cmap=cmap)
        bar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        bar.set_label("Spectral Density")
        ax.set_xlabel('GPS Times')
        ax.set_zlabel('Merger Frequencies')
        ax.set_title('PBH Event Mergers: Shadow Network + Topological Landscape')
        plt.grid(b=None)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.set_xticklabels(xticksl)

        #ax.set_zlim(1,20)
        plt.xticks(rotation=30)
        ax.view_init(30, 140)
        plt.tight_layout()
        plt.show()

    def plot_spectrograms(self):
        figure = plt.gcf()
        figure.clf()
        figure.add_subplot(221)
        figure.add_subplot(222)
        figure.add_subplot(223)
        figure.add_subplot(224)
        axes = figure.get_axes()
        count=0
        figure.suptitle("Spectrogram")
        for k,f in event_names.items():
            ax = axes[count]
            fn = f['file']
            if fn == "":
                return
            t0 = f['ts']
            strain = self.get_strain(fn, f['ts'])
            dt = 1  #-- Set width of q-transform plot, in seconds
            hq = strain.q_transform(outseg=(t0-dt, t0+dt))
            ax.pcolormesh(np.linspace(0,2.4,hq.shape[0]), np.linspace(0,1000,hq.shape[1]), hq.T)
            count+=1
        plt.show()


    def plot_heirarchy(self):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from matplotlib import pyplot as plt
        df = pd.read_csv("ligo_classifications2.csv")

        for i, g in df.groupby('Name of Cluster'):
            X = []
            for j, cluster in g.groupby('Event name '):
                X.append([cluster['Value of Each Node'].mean()])
            Z = linkage(X, 'single')
            fig,ax = plt.subplots()
            dn = dendrogram(Z)
            ax.set_title("Top level event {}".format(i))

        for i, g in df.groupby('Event name '):
            X = []
            for j, cluster in g.groupby('Name of Cluster'):
                X.append([cluster['Value of Each Node'].mean()])
            Z = linkage(X, 'single')
            fig, ax = plt.subplots()
            dn = dendrogram(Z)
            ax.set_title("Cluster {}".format(i))
            for j, cluster in g.groupby('Name of Cluster'):
                X2 = []
                for _, node in cluster.groupby('Nodes in Cluster'):
                    for u,uu in node.iterrows():
                        X2.append([uu['Value of Each Node']])
                Z = linkage(X2, 'single')
                fig, ax = plt.subplots()
                ax.set_title("{}:{}".format(i,j))
                dn = dendrogram(Z)
            break
        plt.show()

    def plot_network(self):
        df = pd.read_csv("ligo_classifications2.csv")
        event_info = {
            'GW170817': {'ts': 1187008882,
                         'pf': 800,
                         'snr': 33.2},
            'GW190814': {'ts': 1249852257,
                         'pf': 960,
                         'snr': 25.6},
            'GW190521': {'ts': 1242442967,
                         'pf': 1800,
                         'snr': 22.4},
            'GW170729': {'ts': 1185389807,
                         'pf': 1240,
                         'snr':10.2
            }
        }

        model = EventScape()
        for event_name, eventdf in df.groupby('Event name '):

            event = RKEvent(name=event_name, attributes=event_info[event_name])
            model.add_event(event)

            for cluster_name, clusterdf in eventdf.groupby('Name of Cluster'):

                rk_cluster = RKCluster(name=event_name)
                event.add_cluster(rk_cluster)
                all_nodes = []
                # add nodes
                for _, nodedf in clusterdf.iterrows():
                    name = nodedf['Nodes in Cluster']
                    att =  {
                        'value': nodedf['Value of Each Node'],
                        'event': nodedf['Event name '],
                        'name_of_cluster': nodedf['Name of Cluster'],
                        'name': name,
                        "ts": event_info[event_name]['ts'],
                        "pf": event_info[event_name]['pf']
                    }


                    #fthresh = [lambda x: 'Q-Value' in x.attributes['name'] and x.value < 0.4, lambda x: 'Spin' in x.attributes['name'] and x.value < 1]
                    nfilter = self.master.toolbar.filters[att['name_of_cluster']] #[lambda x: 'Q-Value' in x.attributes['name'] and x.value < 0.4, lambda x: 'Spin' in x.attributes['name'] and x.value < 1]
                    passes = True

                    if passes:
                        value = att['value']
                        nodes_to_add = 1

                        if nfilter.is_active() and nfilter.sensitivity is not None:
                            n =  int(np.floor((value - nfilter.min()) /  nfilter.sensitivity()))
                            if n < 0:
                                raise ValueError("Error: nodes to add is negative")
                            nodes_to_add += n

                        import uuid
                        for i in range(nodes_to_add):
                            node = Node(label=name + "_" + str(i), id=uuid.uuid4(), value = att['value'], attributes=att)
                            print("Adding node to cluster {}:{}".format(node.label, node.get_id()))
                            rk_cluster.add_node(node)
                            all_nodes.append(node)
                    else:
                        print("Does not pass")

                # add edges. fully connected
                nodes = rk_cluster.get_nodes()
                pairs = set()
                edge_count = 0
                for n1 in nodes:
                    for n2 in nodes:
                        if n1.get_id() == n2.get_id():
                            continue
                        ids = sorted([n1.get_id(), n2.get_id()])
                        pid = "{}".format(ids)
                        if pid in pairs:
                            continue
                        pairs.add(pid)
                        ret = (n1, n2) if n1.value > n2.value else (n2, n1)
                        rk_cluster.add_edge(Edge(n1.get_id(), n2.get_id()))
                        edge_count+=1
                print("RK Cluster has {} edges for {} nodes. Edge count {}. NPairs".format(len(rk_cluster.get_edges()), len(nodes), edge_count, len(pairs)))


        xyz = self.master.toolbar.xyz
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        ax = plt.gca()

        ax.set_xlabel('Timestamps (GPS)')
        ax.set_ylabel('Peak Frequency (Hz)')
        ax.set_zlabel('SNR')
        ax.set_title("RK Diagrams")
        ax.legend()
        angle = 0

        if self.master.toolbar.xyz.use_threshold():
            xticks = np.linspace(1185389807.3, 1449852257.0, 10)
            yticks = np.linspace(0,3000, 10)
            X, Y = np.meshgrid(xticks, yticks)
            thresh = self.master.toolbar.xyz.threshold()
            if thresh is not None:
                L = np.ones(X.shape) * thresh
                ax.plot_wireframe(X, Y, L, color='red')
            else:
                print("Thresh not set")

        ax.view_init(30, angle)
        model.visualize(ax, distance_from_center=self.master.toolbar.xyz.spread(), center_size=self.master.toolbar.xyz.clustersize(), x=xyz.xstatus(), y=xyz.ystatus(), z=xyz.zstatus(), random=True)
        fig.suptitle("Event Scape")

        event_colors = {
            'GW170817': 'blue',
            'GW190814': 'magenta',
            'GW190521': 'cyan',
            'GW170729': 'brown'
        }

        colors = ['magenta', 'blue', 'cyan', 'brown']
        for i, event in enumerate(model.events):
            fig = plt.figure(dpi=100)
            ax = Axes3D(fig)
            ax = plt.gca()
            angle = 0
            ax.view_init(30, angle)
            plt.grid(b=None)
            plt.axis('off')
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            event.visualize(ax, distance_from_center=self.master.toolbar.xyz.spread(), center_color=event_colors[event.name], show_legend=False, random=True, show_minor=True)
            fig.suptitle(event.name)

        plt.show()

    def plot_pair_plots(self):
        '''
        plot secondary measures
        '''
        #import seaborn as sns
        #j
        #df = pd.read_csv("ligo_events.csv")
        #sns.pairplot(df.iloc[:,8:15], dropna=True, corner=True,     plot_kws=dict(marker="+", linewidth=1),  diag_kws=dict(fill=False))
        fig, ax = plt.subplots(dpi=300)
        img = plt.imread("pairplot.png")
        ax.imshow(img)
        ax.axis('off')
        plt.show()


    def plot_rk_diagrams(self):

        df = pd.read_csv("ligo_classifications2.csv")
        event_info = {
            'GW170817': {'ts': 1187008882,
                         'pf': 800,
                         'snr': 33.2},
            'GW190814': {'ts': 1249852257,
                         'pf': 960,
                         'snr': 25.6},
            'GW190521': {'ts': 1242442967,
                         'pf': 1800,
                         'snr': 22.4},
            'GW170729': {'ts': 1185389807,
                         'pf': 1240,
                         'snr':10.2
            }
        }

        model = EventScape()
        for event_name, eventdf in df.groupby('Event name '):

            event = RKEvent(name=event_name, attributes=event_info[event_name])
            model.add_event(event)

            for cluster_name, clusterdf in eventdf.groupby('Name of Cluster'):

                rk_cluster = RKCluster(name=event_name)
                event.add_cluster(rk_cluster)
                all_nodes = []
                # add nodes
                for _, nodedf in clusterdf.iterrows():
                    name = nodedf['Nodes in Cluster']
                    att =  {
                        'value': nodedf['Value of Each Node'],
                        'event': nodedf['Event name '],
                        'name_of_cluster': nodedf['Name of Cluster'],
                        'name': name,
                        "ts": event_info[event_name]['ts'],
                        "pf": event_info[event_name]['pf']
                    }


                    #fthresh = [lambda x: 'Q-Value' in x.attributes['name'] and x.value < 0.4, lambda x: 'Spin' in x.attributes['name'] and x.value < 1]
                    nfilter = self.master.toolbar.filters[att['name_of_cluster']] #[lambda x: 'Q-Value' in x.attributes['name'] and x.value < 0.4, lambda x: 'Spin' in x.attributes['name'] and x.value < 1]
                    passes = True

                    if nfilter.is_active():
                        value = att['value']
                        if nfilter.min() is not None and value < nfilter.min():
                            passes = False
                            break
                        if nfilter.max() is not None and value > nfilter.max():
                            passes = False
                            break

                    if passes:
                        value = att['value']
                        nodes_to_add = 1

                        if nfilter.is_active() and nfilter.sensitivity is not None:
                            n =  int(np.floor((value - nfilter.min()) /  nfilter.sensitivity()))
                            if n < 0:
                                raise ValueError("Error: nodes to add is negative")
                            nodes_to_add += n

                        import uuid
                        for i in range(nodes_to_add):
                            node = Node(label=name + "_" + str(i), id=uuid.uuid4(), value = att['value'], attributes=att)
                            print("Adding node to cluster {}:{}".format(node.label, node.get_id()))
                            rk_cluster.add_node(node)
                            all_nodes.append(node)
                    else:
                        print("Does not pass")

                # add edges. fully connected
                nodes = rk_cluster.get_nodes()
                pairs = set()
                edge_count = 0
                for n1 in nodes:
                    for n2 in nodes:
                        if n1.get_id() == n2.get_id():
                            continue
                        ids = sorted([n1.get_id(), n2.get_id()])
                        pid = "{}".format(ids)
                        if pid in pairs:
                            continue
                        pairs.add(pid)
                        ret = (n1, n2) if n1.value > n2.value else (n2, n1)
                        rk_cluster.add_edge(Edge(n1.get_id(), n2.get_id()))
                        edge_count+=1
                print("RK Cluster has {} edges for {} nodes. Edge count {}. NPairs".format(len(rk_cluster.get_edges()), len(nodes), edge_count, len(pairs)))

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)
                pass

        # case where all 3

        xyz = self.master.toolbar.xyz
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        ax = plt.gca()

        ax.set_xlabel('Timestamps (GPS)')
        ax.set_ylabel('Peak Frequency (Hz)')
        ax.set_zlabel('SNR')
        ax.set_title("RK Diagrams")
        ax.legend()
        angle = 0

        if self.master.toolbar.xyz.use_threshold():
            xticks = np.linspace(1185389807.3, 1449852257.0, 10)
            yticks = np.linspace(0,3000, 10)
            X, Y = np.meshgrid(xticks, yticks)
            thresh = self.master.toolbar.xyz.threshold()
            if thresh is not None:
                L = np.ones(X.shape) * thresh
                ax.plot_wireframe(X, Y, L, color='red')
            else:
                print("Thresh not set")

        ax.view_init(30, angle)
        model.visualize(ax, distance_from_center=self.master.toolbar.xyz.spread(), center_size=self.master.toolbar.xyz.clustersize(), x=xyz.xstatus(), y=xyz.ystatus(), z=xyz.zstatus())
        fig.suptitle("Event Scape")

        event_colors = {
            'GW170817': 'blue',
            'GW190814': 'magenta',
            'GW190521': 'cyan',
            'GW170729': 'brown'
        }

        colors = ['magenta', 'blue', 'cyan', 'brown']
        for i, event in enumerate(model.events):
            fig = plt.figure(dpi=100)
            ax = Axes3D(fig)
            ax = plt.gca()
            angle = 0
            ax.view_init(30, angle)
            plt.grid(b=None)
            plt.axis('off')
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            event.visualize(ax, distance_from_center=self.master.toolbar.xyz.spread(), center_color=event_colors[event.name], show_legend=False)
            fig.suptitle(event.name)

        plt.show()

    def load_data(self):
        for event, data in event_names.items():
            t0 = data['ts']
            detector='H1'
            url = get_urls(detector, t0, t0)[-1]
            print("Trying to find event for {}: {}".format(t0, url))
            fn = os.path.basename(url)
            data['file'] = fn
            if not os.path.exists(fn):
                print('Downloading: ' , url)
                with open(fn,'wb') as strainfile:
                    straindata = requests.get(url)
                    strainfile.write(straindata.content)
            else:
                print("{} is already downloaded".format(event))

class FilterTB(tk.Frame):

    def __init__(self, master=None, name=None):
        super().__init__(master)

        self.activeTgl = tk.IntVar()
        self.label = tk.Label(self, text=name, width=30)
        self.label.grid(column=1, row=0, sticky=tk.NSEW, columnspan=3)
        i=4
        self.minEntry = tk.Entry(self, width=5)
        self.minEntry.grid(column=i, row=0, sticky=tk.NSEW)#pack(side=tk.LEFT)
        i+=1
        self.minlabel = tk.Label(self, text="min")
        self.minlabel.grid(column=i, row=0, sticky=tk.NSEW, columnspan=1)
        i+=1
        self.maxEntry = tk.Entry(self, width=5)
        self.maxEntry.grid(column=i, row=0, sticky=tk.NSEW)#.pack(side=tk.LEFT)
        i+=1
        self.maxlabel = tk.Label(self, text="max")
        self.maxlabel.grid(column=i, row=0, sticky=tk.NSEW, columnspan=1)
        i+=1
        self.sensitivityEntry = tk.Entry(self, width=5)
        self.sensitivityEntry.grid(column=i, row=0, sticky=tk.NSEW)#pack(side=tk.LEFT)
        i+=1
        self.senslabel = tk.Label(self, text="sensitivity")
        self.senslabel.grid(column=i, row=0, sticky=tk.NSEW, columnspan=1)

        i+=1
        self.applyFilter = tk.Checkbutton(self, text="use?", variable=self.activeTgl)
        self.applyFilter.grid(column=i, row=0, sticky=tk.NSEW)#pack(side=tk.LEFT)


    def min(self):
        try:
            return float(self.minEntry.get())
        except:
            return None

    def max(self):
        try:
            return float(self.maxEntry.get())
        except:
            return None

    def sensitivity(self):
        try:
            v =  float(self.sensitivityEntry.get())
            if v >= 0:
                return v
            return None
        except:
            return None

    def is_active(self):
        return self.activeTgl.get()

class XYZTB(tk.Frame):
    def __init__(self, master=None, name=None):
        super().__init__(master)
        count=0
        self.x = tk.IntVar(value=1)
        self.y = tk.IntVar(value=1)
        self.z = tk.IntVar(value=1)
        self.useThresh = tk.IntVar(value=0)
        self.x_btn = tk.Checkbutton(self, text="x", variable=self.x)
        self.x_btn.grid(row=count, column=0)#.pack()
        self.y_btn = tk.Checkbutton(self, text="y", variable=self.y)
        self.y_btn.grid(row=count, column=1)#pack()
        self.z_btn = tk.Checkbutton(self, text="z", variable=self.z)
        self.z_btn.grid(row=count, column=2)#.pack()

        self.v1 = tk.StringVar(self, value='10')
        self.spreadlabel = tk.Label(self, text="Spread", width=30)
        self.spreadlabel.grid(column=0, row=2, sticky=tk.NSEW)
        self.spread_entry = tk.Entry(self, width=5, textvariable=self.v1)
        self.spread_entry.grid(column=1, row=2, sticky=tk.NSEW)#pack(side=tk.LEFT)

        self.v2 = tk.StringVar(self, value='300')
        self.clustersizelabel = tk.Label(self, text="ClusterSize", width=30)
        self.clustersizelabel.grid(column=0, row=3, sticky=tk.NSEW)
        self.clustersize_entry= tk.Entry(self, width=5, textvariable=self.v2)
        self.clustersize_entry.grid(column=1, row=3, sticky=tk.NSEW)#pack(side=tk.LEFT)

        i = 3
        i+=1
        self.thresholdUse = tk.Checkbutton(self, text="show threshold?", variable=self.useThresh)
        self.thresholdUse.grid(column=i, row=0, sticky=tk.NSEW)#pack(side=tk.LEFT)

        i+=1
        self.thresholdlabel = tk.Label(self, text="threshold")
        self.thresholdlabel.grid(column=i, row=0, sticky=tk.NSEW, columnspan=1)

        i+=1
        self.threshEntry = tk.Entry(self, width=5)
        self.threshEntry.grid(column=i, row=0, sticky=tk.NSEW)#pack(side=tk.LEFT)

    def xstatus(self):
        return self.x.get()

    def ystatus(self):
        return self.y.get()

    def zstatus(self):
        return self.z.get()

    def threshold(self):
        try:
            rr = self.threshEntry.get()
            rr = float(rr)
            return rr
        except Exception as e:
            print(e)
            return None

    def use_threshold(self):
        return self.useThresh.get()

    def spread(self):
        try:
            v =  float(self.spread_entry.get())
            if v >= 0:
                return v
            return None
        except:
            return None

    def clustersize(self):
        try:
            v =  float(self.clustersize_entry.get())
            if v >= 0:
                return v
            return None
        except:
            return None

class Toolbar(tk.Frame):
    ''' RK Model '''
    def __init__(self, master=None, visualizer=None):
        super().__init__(master)
        self.master = master
        self.filters = {}
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self, text="load_data", command=self.master.visualizer.load_data)
        self.load_button.grid(sticky=tk.NSEW, row=0)#pack(fill=tk.BOTH)
        self.strain_button = tk.Button(self, text="view strain", command=self.master.visualizer.plot_strain)
        self.strain_button.grid(sticky=tk.NSEW, row=1)#pack(fill=tk.BOTH)
        self.asd_button = tk.Button(self, text="view asd", command=self.master.visualizer.plot_asd)
        self.asd_button.grid(sticky=tk.NSEW, row=2)#pack(fill=tk.BOTH)
        self.spectrogram_button = tk.Button(self, text="plot spectrograms", command=self.master.visualizer.plot_spectrograms)
        self.spectrogram_button.grid(sticky=tk.NSEW, row=3)#pack(fill=tk.BOTH)
        self.eventscape_button = tk.Button(self, text="plot event scape", command=self.master.visualizer.plot_eventscape)
        self.eventscape_button.grid(sticky=tk.NSEW, row=4)#pack(fill=tk.BOTH)
        self.filteredscape_button = tk.Button(self, text="plot filtered event scape", command=self.master.visualizer.plot_filteredscape)
        self.filteredscape_button.grid(sticky=tk.NSEW, row=5)#pack(fill=tk.BOTH)
        self.pair_button = tk.Button(self, text="plot pair plot", command=self.master.visualizer.plot_pair_plots)
        self.pair_button.grid(sticky=tk.NSEW, row=6)#pack(fill=tk.BOTH)
        self.pair_button = tk.Button(self, text="plot heiarchy", command=self.master.visualizer.plot_heirarchy)
        self.pair_button.grid(sticky=tk.NSEW, row=7)#pack(fill=tk.BOTH)
        self.pair_network = tk.Button(self, text="plot network", command=self.master.visualizer.plot_network)
        self.pair_network.grid(sticky=tk.NSEW, row=8)#pack(fill=tk.BOTH)
        self.rk_diagram_button = tk.Button(self, text="plot rk diagrams", command=self.master.visualizer.plot_rk_diagrams)
        self.rk_diagram_button.grid(row=9, sticky=tk.NSEW)#pack(fill=tk.BOTH)
        df = pd.read_csv("ligo_classifications2.csv")
        filters = df['Name of Cluster'].unique().tolist()
        count = 9
        count+=1
        for f in filters:
            tb = FilterTB(master=self, name=f)
            tb.grid(row=count, sticky=tk.NSEW)
            self.filters[f] = tb
            count+=1
        self.xyz = XYZTB(master=self)
        self.xyz.grid(row=count, sticky=tk.NSEW)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

def main():
    app = RKModelApplication(master=root)
    app.master.title("RK-Diagram Demo")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
