import json
import numpy as np
from braket.aws import AwsDevice
from braket.ocean_plugin import BraketSampler, BraketDWaveSampler
import matplotlib.pyplot as plt
# magic word for producing visualizations in notebook
import networkx as nx
from collections import defaultdict
from itertools import combinations
#import math
import time
import dwave_networkx as dnx
import dimod #.binary_quadratic_model import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
import plotly.graph_objects as go

def build_graph(int_array,scale=1,seed=np.random):
    int_array = np.sort(int_array)
    #graph = nx.Graph()
    #random_pos = nx.random_layout(graph, seed=42)
    G = nx.random_geometric_graph(len(int_array), radius=0.125, dim=2, pos=None, p=2, seed=seed)
    for ix, n in enumerate(int_array):
        for m in int_array[ix:]:
            if m==n:
                G.add_edge(n,n,weight=scale*n**2)
            else:
                G.add_edge(n,m,weight=2*scale*n*m)
    return G

def visualize_graph(G,colorscale='YlGnBu'):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(x=edge_x, y=edge_y,line=dict(width=0.5, color='#888'),hoverinfo='none',mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y,mode='markers',hoverinfo='text',marker=dict(showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale=colorscale,reversescale=True,color=[],size=10,
        colorbar=dict(thickness=15,title='Node value',xanchor='left',titleside='right'),line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        #node_text.append('# of connections: '+str(len(adjacencies[1])))
        node_text.append('Value of node: '+str(list(G.nodes())[node]))
        #print("node: ",node)
    #node_trace.marker.color = node_adjacencies
    node_trace.marker.color = list(G.nodes)
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig
#fig.show()
def get_quadratic(int_array,scale=1):
    int_array = np.sort(int_array)
    quadratic = {(i,j): scale*2*n_i*n_j for j, n_j in enumerate(int_array) for i, n_i in enumerate(int_array) if j>i}
    offset = scale*np.sum(int_array**2)
    return offset, quadratic 

def sample(ham,sampler=None, sampling_method="sample", ascending=True,sample_kw={}):
    if sampler is None or sampler=="exact":
        sampler = dimod.ExactSolver()
    if sampling_method == "sample": 
        df = sampler.sample(ham,**sample_kw).to_pandas_dataframe() 
    elif sampling_method == "qubo":
        df = sampler.sample_qubo(ham,**sample_kw).to_pandas_dataframe() 
    elif sampling_method == "ising":
        df = sampler.sample_ising(ham,**sample_kw).to_pandas_dataframe() 
    df.sort_values(by=['energy'], axis=0,ascending=ascending,inplace=True)
    return df

def plot_energy_distribution(samples_df,subplots_kw={"figsize":(20,8)},
                             hist_kw={"bins":25,"histtype":"bar","color":"blue","ec":"black"}):
    energies=samples_df.energy.values
    fig, ax = plt.subplots(**subplots_kw)
    ax.hist(energies,**hist_kw)