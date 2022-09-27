#!/usr/bin/env python
import matplotlib.pyplot as plt

class RKLandscapeVisualization:
    '''
    Visualization Helpers for showing Eventscape in 3d
    on graph
    '''

    def plot_network(G, angle, save):

        pos = nx.get_node_attributes(G, 'pos')  # Get number of nodes
        n = G.number_of_nodes()

        with plt.style.context(('default')):

            fig = plt.figure(figsize=(10,7))
            ax = Axes3D(fig)

            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            for key, value in pos.items():
                value = value[0]
                xi = value[0]
                yi = value[1]
                zi = value[2]

                s = 5+10*G.degree(key)
                if 'Central Node' in key:
                    s = 300
                    ax.scatter(xi, yi, zi, s=s, edgecolors='k', c=G.nodes()[key]['color'], alpha=0.7)

                ax.scatter(xi, yi, zi, s=s, edgecolors='k', c=G.nodes()[key]['color'], alpha=0.7)

            for i,j in enumerate(G.edges()):
                fr = j[0]
                to = j[1]
                x = np.array((pos[fr][0][0], pos[to][0][0]))
                y = np.array((pos[fr][0][1], pos[to][0][1]))
                z = np.array((pos[fr][0][2], pos[to][0][2]))
                ax.plot(x, y, z, c='black', alpha=0.5)

        ax.view_init(30, angle)
        ax.yaxis.grid(True, which='major')
        ax.xaxis.grid(True, which='major')
        ax.set_facecolor('white')
        ax.grid()
        ax.set_title("R-K Diagrams", size=20)


    def add_central_node(G):
        pos = nx.get_node_attributes(G, 'pos')
        posit = [x[1][0] for x in pos.items()]
        central_node = [np.mean(posit, axis=0)]

        G.add_nodes_from([('Central Node', {
            'value': 10,
            'event': 'Central Node',
            'name_of_cluster': 'Central',
            'color': 'black',
            'pos': central_node
        })])


        closest_nodes = {}
        for key in G.nodes():
            pp = G.nodes()[key]['pos']
            cluster = G.nodes()[key]['name_of_cluster']
            dist = np.linalg.norm(np.array(central_node)-np.array(pp))
            if cluster not in closest_nodes:
                closest_nodes[cluster] = (key, dist)
            elif dist < closest_nodes[cluster][1]:
                closest_nodes[cluster] = (key, dist)

        for k, edge in closest_nodes.items():
            G.add_edges_from([(edge[0], 'Central Node')])
        return G

def plot_strain_data(fn, ax, t0):
    # -- Read strain data
    strain = TimeSeries.read(fn,format='hdf5.losc')
    center = int(t0)
    strain = strain.crop(center-16, center+16)
    ax.plot(strain)
    return strain

def plot_asd(strain, ax):
    # -- Plot ASD
    fig2 = strain.asd(fftlength=8)
    ax.plot(fig2)
    ax.set_xlim(10,2000)
    ax.set_ylim(1e-24, 1e-19)
    ax.set_yscale('log')
    ax.set_xscale('log')

def plot_whitened(strain,t0, ax):
    white_data = strain.whiten()
    bp_data = white_data.bandpass(30, 400)
    ax.plot(bp_data)
    ax.set_xlim(t0-0.2, t0+0.1)
    return bp_data

def plot_spectrogram(strain, t0, dt=1, ax=None):
    dt = 1  #-- Set width of q-transform plot, in seconds
    hq = strain.q_transform(outseg=(t0-dt, t0+dt))
    ax.pcolormesh(np.linspace(0,2.4,hq.shape[0]), np.linspace(0,1000,hq.shape[1]), hq.T)
    #ax.set_ylim(0.1,3)
    #ax.set_yscale('log')
    return hq
