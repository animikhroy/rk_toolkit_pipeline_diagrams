#!/usr/bin/env python3
from networkx.classes.digraph import DiGraph
from rktoolkit.functions.distance import mahalanobis, jaccard
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib
from common import *
import itertools
import time
import nevergrad as ng
from rktoolkit.visualizers.util import draw_rk_diagram
from rktoolkit.ml.objective_functions import SampleObjectiveFunction

def visualize_n(graphs, n=1, *args, **kwargs):
    fig, ax = plt.subplots(2,int(n/2), figsize=(18,10))
    for i in range(n):
        j, k = i % int(n/2), int(i/(n/2))
        draw_rk_diagram(graphs[i], ax=ax[k][j], *args, **kwargs)
        ax[k][j].set_title(graphs[i].label)

def make_n_rkmodels(df, labels, hft, pipeline, n=1, indexes=None):
    rkmodels = []

    if indexes is not None:
        for i in indexes:
            base = hft.transform(df.iloc[i])
            rkm = pipeline.transform(base, is_base=False)
            rkm.label = labels.iloc[i]["id (Event ID)"]
            rkm.class_label = labels.iloc[i]["Classification"]
            rkm.id = i
            rkmodels.append(rkm)
        return rkmodels

    for i, row in df.iterrows():
        if i > n-1:
            break
        base = hft.transform(df.iloc[i])
        rkm = pipeline.transform(base, is_base=False)
        rkm.class_label = labels.iloc[i]["Classification"]
        rkm.label = labels.iloc[i]["id (Event ID)"]
        rkm.id = i
        rkmodels.append(rkm)

    return rkmodels

def compute_distance(df):
    mdist = mahalanobis(df, df)
    return (mdist - mdist.min()) / (mdist.max() - mdist.min())

def compute_distances(rkmodels, mdist, w=[.5, .5]):
    distances = []
    w = np.array(w)
    w = w / w.sum()
    for p in itertools.combinations(rkmodels, 2):
        tdist = p[0].get().edge_distance(p[1].get())
        vdist = np.sqrt(np.power(mdist[p[0].id] - mdist[p[1].id], 2))
        if tdist < 0 or vdist < 0:
            raise ValueError("Less than 0 for dist: {}: {}".fromat(tdist, vdist))
        distances.append(np.array([tdist, vdist]) * w)
    return distances

def train(param_size, pipeline, df, hft, mdist, iterations=2000, batch_size=10):
    loss_history = []
    w0 = pipeline.get_w()
    optimizer = ng.optimizers.NGOpt(parametrization=param_size, budget=iterations, num_workers=5)
    prev_time = None
    start = time.time()
    ofunc = SampleObjectiveFunction(pipeline, batch_size, w0, df, hft, mdist, compute_distances)
    for i in range(optimizer.budget):
        x = optimizer.ask()
        loss = ofunc.evaluate(*x.args, **x.kwargs)
        if i % 100 == 0:
            now = time.time()
            if prev_time is not None:
                print("Iteration: {:05d}. Loss: {:.08f}. ITime {:.02f} seconds. Total time: {:.02f}".format(i, loss, now - prev_time, now-start))
            prev_time = now
        loss_history.append(loss)
        optimizer.tell(x, loss)
    recommendation = optimizer.provide_recommendation()
    return recommendation.value, loss_history

def extract_rk_features(rk):
    rk = rk.get()
    return [len(rk.nodes),
            len(rk.edges),
            nx.average_node_connectivity(rk),
            nx.edge_connectivity(rk)]

def show(trained_weights, loss_history, pipeline, df, hft, w0):
    rkmodels = []
    pupdate = pipeline.remap(trained_weights, w0[1])
    for i, row in df.iterrows():
        g = hft.transform(row)
        g = pupdate.transform(g)
        g.id = i
        rkmodels.append(g)
        if i > 10:
            break

    fig, ax = plt.subplots(figsize=(12,5))
    ax.set_title("Loss Over Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.plot(loss_history)
    plt.show()
    plt.savefig("rendered/loss_over_time.png",  dpi = 300, bbox_inches="tight")
