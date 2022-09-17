from networkx.dendrogram import hierarchy_pos
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

def draw_rk_diagram(rkmodel, ax=None, with_labels=True, minsize=100, center_color='green',
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
