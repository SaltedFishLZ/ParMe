import os
import sys

if sys.version <= "3.7":
    try:
        from collections.abc import OrderedDict
    except ImportError:
        from collections import OrderedDict
else:
    OrderedDict = dict

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt 

from EleNetX.visualize import plot_ele_nx

import utils


def plot_input(G: nx.Graph,
               out_dir: str,
               name: str) -> None:

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    pos = nx.nx_agraph.graphviz_layout(G)

    plot_ele_nx(G, ax,
                pos=pos,
                node_color_keyword="module")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'input.{}.png'.format(name)))
    plt.savefig(os.path.join(out_dir, 'input.{}.pdf'.format(name)))

    plt.clf(); plt.cla(); plt.close()


def draw_cut_edges(ax:mpl.axes.Axes,
                   G: nx.Graph,
                   X: np.ndarray,
                   pos):
    
    sgids = utils.onehot_to_index(X, axis=1)

    for i, v in enumerate(G.nodes):
        G.nodes[v]['id'] = i
    
    cut_edges = []
    for e in G.edges:
        v1, v2 = e
        k1 = G.nodes[v1]['id']; k2 = G.nodes[v2]['id']
        j1 = sgids[k1]; j2 = sgids[k2]
        if (j1 != j2):
            cut_edges.append(e)
    
    all_edge_weights = [G.edges[e]['weight'] for e in G.edges]
    cut_edge_weights = [G.edges[e]['weight'] for e in cut_edges]

    alpha = np.mean(all_edge_weights)
    print(alpha)
    print(cut_edge_weights / alpha)

    nx.draw_networkx_edges(ax=ax, G=G, pos=pos,
                           edgelist=cut_edges,
                           width=np.log(cut_edge_weights) / np.log(alpha),
                           style=':')

    return ax


def get_subgraphs(G, X):
    l, m = X.shape
    l_ = len(G); assert l_ == l

    for i, v in enumerate(G.nodes):
        G.nodes[v]['id'] = i

    sgids = utils.onehot_to_index(X, axis=1)
    sG_v_map = dict.fromkeys(range(m))
    # must init dict val with new objects one by one
    for j in sG_v_map:
        sG_v_map[j] = []
    for v in G.nodes:
        j = sgids[G.nodes[v]['id'] ]
        # print('v = ', v, 'j = ', j)
        sG_v_map[j].append(v)
        # print(sG_v_map[j])
    
    sGs = []
    for j in range(m):
        nodes = sG_v_map[j]
        # print(nodes)
        sG = G.subgraph(nodes)
        sGs.append(sG)
        # print(sG)

    return sGs


def plot_output(G: nx.Graph,
                X: np.ndarray,
                out_dir: str,
                name: str,
                figsize=(10, 10)) -> None:
    """
    :param G:
    :param X: $l \times m$
    """
    l = len(G.nodes)
    l_, m = X.shape
    assert l_ == l, ValueError('shape mismatch')

    sgids = utils.onehot_to_index(X, axis=1)

    # print('=' * 64)
    # print('# nodes: ', l)
    # print('# subgraphs:', m)
    # print('cluster id for nodes in ', name)
    # print(sgids)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    pos = nx.nx_agraph.graphviz_layout(G)

    nx.draw_networkx(G, ax=ax, pos=pos,
                     node_color=sgids,
                     cmap=plt.cm.magma, 
                     with_labels=True)

    # draw_cut_edges(ax=ax, G=G, X=X, pos=pos)

    # sGs = get_subgraphs(G=G, X=X)
    # for j, sG in enumerate(sGs):
    #     print('j = ', j, 'l = ', len(sG))

    #     all_edge_weights = [G.edges[e]['weight'] for e in G.edges]
    #     sub_edge_weights = [G.edges[e]['weight'] for e in sG.edges]
    #     alpha = np.median(all_edge_weights)
    #     normalized_sub_edge_widths = sub_edge_weights / alpha
        
    #     nx.draw_networkx(ax=ax, G=sG, pos=pos, width=np.power(normalized_sub_edge_widths, 0.3) )

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'output.{}.png'.format(name)))
    plt.savefig(os.path.join(out_dir, 'output.{}.pdf'.format(name)))

    plt.clf(); plt.cla(); plt.close()