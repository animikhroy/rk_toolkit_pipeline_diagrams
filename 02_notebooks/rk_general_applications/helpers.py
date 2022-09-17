import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def make_df_from_sheet(file):
    sheets = pd.ExcelFile(file).sheet_names
    data = {}
    for s in sheets:
        key = s.strip()
        if key == "Prodcut":
            key = "Product"
        data[key] = pd.read_excel(file, sheet_name=s)
        
    for k, v in data.items():
        cols = v.columns.values
        for i, c in enumerate(v.columns):
            cols[i] = c.split("(")[0].strip()
        v.columns = cols
        
    return data

def preprocess(data):
    df = pd.concat([data['Identity'], data['Order'], data['Sales'], data['Product'], data['Location']], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df["Categories"] = OneHotEncoder().fit_transform(df[["Category"]]).toarray().astype(int).tolist()
    df["SubCategories"] = OneHotEncoder().fit_transform(df[["Sub-Category"]]).toarray().astype(int).tolist()
    df["Ship Mode Enc"] = OneHotEncoder().fit_transform(df[["Ship Mode"]]).toarray().astype(int).tolist()

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
        category_embedding = combine(v["Categories"])
        subcategory_embedding = combine(v["SubCategories"])
        ship_mode_embedding = combine(v["Ship Mode Enc"])

        returns_total = 0
        for v in v["Returns"]:
            if v == "Yes":
                returns_total+=1

        row = [k, total_sales_volume, total_discount, total_quantity, total_profit, postal_code, category_embedding, subcategory_embedding, returns_total, ship_mode_embedding, country, state, region, city]
        rows.append(row)

    pdf = pd.DataFrame(rows, columns=columns)
    return pdf


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

def draw_g(G, ax=None, with_labels=True, minsize=100, alpha=300, emult=2, make_axis=False, width=2*math.pi):
    if make_axis:
        fig, ax = plt.subplots(figsize=(10, 10))
    if ax is None:
        ax = plt.gca()
        
    pos = hierarchy_pos(G, 'root', width = width, xcenter=0)
    new_pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    cols = [ n[1].get('color', 'black') for n in list(G.nodes.items())]
    sizes = []
    for n in list(G.nodes.items()):
        v = n[1].get("value",1)
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
    nx.draw(G, pos=new_pos, with_labels=with_labels, font_size=10, node_size=sizes, ax=ax, node_color = cols)
    nx.draw_networkx_nodes(G, pos=new_pos, nodelist = ['root'], node_color = 'green', ax=ax, node_size = sizes.max()*emult)
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
def visualize_10(graphs, *args, **kwargs):
    fig, ax = plt.subplots(2,5, figsize=(18,5))
    for i in range(10):
        j, k = i % 5, int(i/5)
        draw_g(graphs[i], ax=ax[k][j], *args, **kwargs)
        ax[k][j].set_title(i)


import itertools
def compute_distances(graphs, mdist, w=[.5, .5]):
    distances = []
    w = np.array(w)
    w = w / w.sum()
    for p in itertools.combinations(graphs, 2):
        tdist = p[0].edge_distance(p[1])
        vdist = np.sqrt(np.power(mdist[p[0].id] - mdist[p[1].id], 2))
        if tdist < 0 or vdist < 0:
            raise ValueError("Less than 0 for dist: {}: {}".fromat(tdist, vdist))
        distances.append(np.array([tdist, vdist]) * w)
    return distances


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


