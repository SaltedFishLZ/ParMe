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

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G)

    plot_ele_nx(G, ax,
                pos=pos,
                node_color_keyword="module")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'input.{}.png'.format(name)))
    plt.savefig(os.path.join(out_dir, 'input.{}.pdf'.format(name)))

    plt.clf(); plt.cla(); plt.close()


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

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G)
    
    nx.draw_networkx(G, ax=ax, pos=pos,
                     node_color=sgids,
                     cmap=plt.cm.magma, 
                     with_labels=True)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'output.{}.png'.format(name)))
    plt.savefig(os.path.join(out_dir, 'output.{}.pdf'.format(name)))

    plt.clf(); plt.cla(); plt.close()