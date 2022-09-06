import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import nevergrad as ng
from concurrent import futures
import time



def make_df_from_sheet(file):
    return pd.read_csv(file)

def preprocess(df):
    '''
    This is a basic preprocessor. 
    There may be much better preprocessors built later.
    '''
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # only use numeric types
    df["Classification"] = df["Classification"].str.strip() # strip whitespaces

    # some specific columns to handle
    watched_columns = ["p_astro","SNR_network_matched_filter", "luminosity_distance", "far","chirp_mass_source"]
    for c in watched_columns:
        df = df.fillna(0)
        
    y = df["Classification"] # the label for classification
    ids = df["id (Event ID)"] # the ids we will use

    df.drop(['far_lower', 'far_upper','p_astro_lower','p_astro_upper'], inplace=True, axis=1) # these were bad columns

    subdf = df.select_dtypes(include=numerics) # only take numeric data for processing
    invalid_columns = subdf.isna().any()[subdf.isna().any().values== True].index 
    subdf = subdf.drop(invalid_columns, axis=1) 
    subdf = subdf.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    ids = ids.reset_index(drop=True)
    return subdf, y, df, ids


# +
import matplotlib
from common import *

class BaseOntologyTransform():
    
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
# +
from rktoolkit.visualizers.networkx.dendrogram import hierarchy_pos
import math
import numbers
import matplotlib.pyplot as plt 

def draw_graph(G, ax=None, with_labels=True, minsize=100, 
                    alpha=300, emult=2, make_axis=False, width=2*math.pi):
    
    if make_axis:
        fig, ax = plt.subplots(figsize=(10,10))

    if ax is None:
        ax = plt.gca()
        
    pos = hierarchy_pos(G, 'root', width = width, xcenter=0)
    pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    colors = [ n[1].get('color', 'black') for n in list(G.nodes.items())]
    
    sizes = []
    for n in list(G.nodes.items()):
        v = n[1].get("value", 1)
        if not isinstance(v, numbers.Number):
            v = 1
        v+=1
        sizes.append(v)
    
    sizes = np.array(sizes)
    sizes = sizes ** 5 #np.exp(sizes)
    sizes = (sizes - sizes.min()) / (sizes.max()-sizes.min())
    sizes *= alpha
    sizes += minsize
    sizes = np.where(np.isnan(sizes), minsize, sizes)
    nx.draw(G, pos=pos, with_labels=with_labels,
            font_size=10, node_size=sizes, ax=ax, node_color = colors, edgecolors = 'black')
    nx.draw_networkx_nodes(G, pos=pos, nodelist = ['root'],
                           node_color = 'green', ax=ax, node_size = sizes.max()*emult)

def draw_rk_diagram(rkmodel, ax=None, with_labels=True, minsize=100, center=[0,0],
                    alpha=300, emult=2, make_axis=False, width=2*math.pi):
    if make_axis:
        fig, ax = plt.subplots(figsize=(10,10))

    if ax is None:
        ax = plt.gca()
    
    # structural pos    
    structural_pos = hierarchy_pos(rkmodel.G, 'root', width = width)
  
    structural_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in structural_pos.items()}  
    for k, pos in structural_pos.items():
        structural_pos[k] = [pos[0] + center[0], pos[1] + center[1]]
   
    structural_colors = [ n[1].get('color', 'black') for n in list(rkmodel.G.nodes.items())]
    
    # filter nodes
    rkgraph = rkmodel.get()
    filtered_pos = {k: structural_pos[k] for k in list(rkgraph.nodes)}
    
    # get indexes 
    arr, nodes = [], list(rkgraph.nodes)
    for i, n in enumerate(list(rkmodel.G.nodes)):
        if n in nodes:
            arr.append(i)
            
    filtered_colors = [structural_colors[i] for i in arr]
    
    sizes = []
    for n in list(rkgraph.nodes.items()):
        v = n[1].get("value", 1)
        if not isinstance(v, numbers.Number):
            v = 1
        v+=1
        sizes.append(v)
    
    sizes = np.array(sizes)
    sizes = sizes ** 5 #np.exp(sizes)
    sizes = (sizes - sizes.min()) / (sizes.max()-sizes.min())
    sizes *= alpha
    sizes += minsize
    sizes = np.where(np.isnan(sizes), minsize, sizes)
    nx.draw(rkgraph, pos=filtered_pos, with_labels=with_labels, 
            font_size=10, node_size=sizes, ax=ax, node_color = filtered_colors, edgecolors = 'black')
    nx.draw_networkx_nodes(rkgraph, pos=filtered_pos, nodelist = ['root'],
                           node_color = 'green', ax=ax, node_size = sizes.max()*emult)
# +
import scipy as sp

def mahalanobis(x=None, data=None, cov=None):
    """
    Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


# -
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


# +
import itertools

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


# -

class RangeFilter():
    
    def __init__(self, min:float=0, max:float=1):
        self.min = min
        self.max = max
        
    def get_knobs(self):
        return {
            "min": self.min,
            "max": self.max
        }
    
    def set_knob(self, knb, v):
        if knb == "min":
            self.min = v
        elif knb == "max":
            self.max = v
        else:
            raise ValueError("No knob {}".format(knb))
    
    def filter(self, node):
        if not "value" in node or node["value"] is None:
            print("Warning: No value present for {}. Could not filter".format(node))
            return False
        
        return not (node["value"] > self.min and node["value"] <= self.max)


class FilterAll():
    
        
    def get_knobs(self):
        pass
    
    def set_knob(self, knb, v):
        pass
    
    def filter(self, node):
        return false


def preprocess2(data):
    df = pd.concat([data['Identity'], data['Order'], data['Sales'], data['Product'], data['Location']], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df["Categories"] = OneHotEncoder().fit_transform(df[["Category"]]).toarray().astype(int).tolist()
    df["SubCategories"] = OneHotEncoder().fit_transform(df[["Sub-Category"]]).toarray().astype(int).tolist()
    df["Ship Mode Enc"] = OneHotEncoder().fit_transform(df[["Ship Mode"]]).toarray().astype(int).tolist()
    df["Region"] = OneHotEncoder().fit_transform(df[["Region"]]).toarray().astype(int).tolist()
    df["State"] = OneHotEncoder().fit_transform(df[["State"]]).toarray().astype(int).tolist()

    rows = []

    def combine(rows):
        cat_embedding = None
        for row in rows:
            if cat_embedding is None:
                cat_embedding = np.array(row).astype(int)
            else:
                cat_embedding = np.add(cat_embedding, np.array(row).astype(int))
        return cat_embedding

    columns = ["order_id", "total_sales", "discount", "total_quantity", "total_profit", "postal_code", "cateogry_embedding", "subcategory_embedding", "returns", "ship_mode_embedding", "country", "state", "region", "city"]

    for k, v in df.groupby(['Order ID']):
        total_sales_volume = v["Sales"].sum()
        total_quantity = v["Quantity"].sum()
        total_profit = v["Profit"].sum()
        postal_code = v["Postal Code"].iloc[0]
        country = v["Country"].iloc[0]
        state = v["State"].iloc[0]
        region = v["Region"].iloc[0]
        city = v["City"].iloc[0]
        total_discount = v["Discount"].sum()

        cat_embedding = None
        category_embedding = combine(v["Categories"]).tolist()
        subcategory_embedding = combine(v["SubCategories"]).tolist()
        ship_mode_embedding = combine(v["Ship Mode Enc"]).tolist()

        returns_total = 0
        for v in v["Returns"]:
            if v == "Yes":
                returns_total+=1

        row = [k, total_sales_volume, total_discount, total_quantity, total_profit, postal_code, category_embedding, subcategory_embedding, returns_total, ship_mode_embedding, country, state, region, city]
        rows.append(row)

    pdf = pd.DataFrame(rows, columns=columns)
    return pdf


# +
import numbers

class SimpleChildLinker():
    
    def __init__(self, theta=1):
        self.theta = theta
        
    def get_knobs(self):
        return {"theta": self.theta}
    
    def set_knob(self, knb, v):
        if knb == "theta":
            self.theta = v
        else:
            raise ValueError("No knob {}".format(knb))
   
    def check_valid_node(self, node) -> bool:
        if "value" not in node:
            return False   
        if not isinstance(node["value"], numbers.Number):
            return False
        return True
    
    def link(self, G):
        gC = copy.deepcopy(G)
        for n in G.nodes:
            for p in itertools.combinations(G.get_children(n), 2):
                if len(p) < 2:
                    continue
                if not self.check_valid_node(G.nodes[p[0]]) or not self.check_valid_node(G.nodes[p[1]]):
                    continue  
                u_v, v_v = G.nodes[p[0]]["value"], G.nodes[p[1]]["value"]
                d = np.linalg.norm(u_v - v_v)
                if d < self.theta:
                    fn = 0 if u_v < v_v else 1
                    tn = 1 ^ fn  
                    gC.add_edge(Edge(u=p[fn], v=p[tn], attributes={"edge_distance": d}))
        return gC


# -
class BaseOntologyTransform():
    
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


def draw_rk_diagram_v2(rkmodel, ax=None, with_labels=True, minsize=100, center_color='green', 
                    alpha=300, emult=2, make_axis=False, width=2*math.pi, xoff=0, yoff=0):
    if make_axis:
        fig, ax = plt.subplots(figsize=(10,10))

    if ax is None:
        ax = plt.gca()
    
    # structural pos    
    structural_pos = hierarchy_pos(rkmodel.G, 'root', width = width, xcenter=0)
    structural_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in structural_pos.items()}   
    structural_colors = [ n[1].get('color', 'black') for n in list(rkmodel.G.nodes.items())]
    
    # filter nodes
    rkgraph = rkmodel.get()
    filtered_pos = {k: structural_pos[k]  for k in list(rkgraph.nodes)}
    
    for k,v in filtered_pos.items():
        filtered_pos[k] = [v[0]+xoff, v[1]+yoff]
        
    # get indexes 
    arr, nodes = [], list(rkgraph.nodes)
    for i, n in enumerate(list(rkmodel.G.nodes)):
        if n in nodes:
            arr.append(i)
            
    filtered_colors = [structural_colors[i] for i in arr]
    
    sizes = []
    for n in list(rkgraph.nodes.items()):
        v = n[1].get("value", 1)
        if not isinstance(v, numbers.Number):
            v = 1
        v+=1
        sizes.append(v)
    
    sizes = np.array(sizes)
    sizes = sizes ** 5 #np.exp(sizes)
    sizes = (sizes - sizes.min()) / (sizes.max()-sizes.min())
    sizes *= alpha
    sizes += minsize
    sizes = np.where(np.isnan(sizes), minsize, sizes)
    nx.draw(rkgraph, pos=filtered_pos, with_labels=with_labels, 
            font_size=10, node_size=sizes, ax=ax, node_color = filtered_colors, edgecolors = 'black')
    nx.draw_networkx_nodes(rkgraph, pos=filtered_pos, nodelist = ['root'],
                           node_color = center_color, ax=ax, node_size = sizes.max()*emult)


class ObjectiveFunction():
    
    def __init__(self, pipeline, sample_size, w0, df, hft, mdist):
        self.pipeline = pipeline
        self.sample_size = sample_size
        self.w0 = w0
        self.df = df
        self.hft = hft
        self.mdist = mdist
        
    def evaluate(self, w):
        pupdate = self.pipeline.remap(w, self.w0[1])
        models = []
        sample = self.df.sample(n=self.sample_size)
        for i, row in sample.iterrows():
            g = self.hft.transform(row)
            g = pupdate.transform(g)
            g.id = i
            models.append(g)
 
        distances = compute_distances(models, self.mdist, [1, 0])
        c = 1 - np.mean(distances)
        return c


def train(param_size, pipeline, df, hft, mdist, iterations=2000, batch_size=10):
    loss_history = []
    w0 = pipeline.get_w()
    optimizer = ng.optimizers.NGOpt(parametrization=param_size, budget=iterations, num_workers=5)
    prev_time = None
    start = time.time()
    ofunc = ObjectiveFunction(pipeline, batch_size, w0, df, hft, mdist)
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
    
