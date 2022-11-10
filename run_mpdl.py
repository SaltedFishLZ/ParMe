# experiment script for MPDL benchmark
import os
import sys
import time
import pickle
import logging
import calendar
from typing import List, Dict

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
import gurobipy as gp
from gurobipy import GRB
import matplotlib as mpl
import matplotlib.pyplot as plt 

import EleNetX.mpdl as mpdl
from EleNetX.utils import obj_attr_cat_to_int
from EleNetX.visualize import plot_ele_nx

from utils import *
from parme import get_parme_model

# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.ParMe.{}.log".format(timestamp)


def load_graphs() -> Dict[str, nx.Graph]:
    """
    """
    graphs: OrderedDict[str, nx.Graph] = OrderedDict()
    # load all benchmark graphs
    for neural_net in mpdl.MPDL_BENCHMARKS.keys():
        num_cfg = mpdl.MPDL_BENCHMARKS[neural_net]['num_cfg']
        # enumerate configs
        for i in range(num_cfg):
            gpath = mpdl.compose_graph_path(neural_net, i)
            G = nx.read_gpickle(gpath)

            mpdl.assign_modules(G)

            name = neural_net + ':' + str(i)
            graphs[name] = G
    
    return graphs


def get_module_index(Gs: Dict[str, nx.Graph],
                     key: str = "module") -> Dict:
    # we must consider all graphs to generate module type index
    # in case some graphs won't include all types of modules
    names = Gs.keys(); graphs = Gs.values()
    G = nx.union_all(graphs, rename=[name + ":" for name in names])
    # get module indices
    module_indices = obj_attr_cat_to_int(G.nodes(), key)
    return module_indices


def get_R0(G: nx.Graph,
           module_indices: OrderedDict,
           key: str = "module") -> np.ndarray:
    """
    """
    l = len(G.nodes); r = len(module_indices)
    R0 = np.zeros((r, l), dtype=int)
    for j, v in enumerate(G.nodes):
        k = module_indices[G.nodes[v][key]]
        R0[k, j] = 1
    return R0


def plot_input(G, name):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G)

    plot_ele_nx(G, ax,
                pos=pos,
                node_color_keyword="module")

    dir_path = os.path.join('figures', 'mpdl')
    os.makedirs(dir_path, exist_ok=True)
    png_path = os.path.join(dir_path, '{}.png'.format(name))
    plt.savefig(png_path)


if __name__ == "__main__":
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    np.set_printoptions(edgeitems=30, linewidth=200)

    Gs = load_graphs()

    module_indices = get_module_index(Gs)

    # init inputs for ParMe
    n = len(Gs); r = len(module_indices)
    Ls = []; R0s = []
    w0 = np.ones(r)
    q = np.asarray([1, 1, 1])
    t = 4; theta = 0.0
    min_size = 8
    max_size = 16
    # update inputs for ParMe
    for name in Gs:
        G = Gs[name]
        L = nx.laplacian_matrix(G).tocoo()
        Ls.append(L)
        R0 = get_R0(G, module_indices)
        R0s.append(R0)

        print("=" * 64)
        print(name)
        print(R0.shape)
        # print(R0)

        plot_input(G, name)


    # model, grb_vars, grb_exprs = get_parme_model(Ls=Ls, q=q, R0s=R0s, w0=w0, t=t,
    #                                              min_size=min_size, max_size=max_size,
    #                                              theta=theta)
    # # update model and Gurobi configuration
    # model.update()
    # model.setParam("LogFile", log_file)
    # model.setParam("LogToConsole", 0)
    # model.setParam('TimeLimit', 2 * 60)

    # model.optimize()

    # (Xs, Zs, R_sup, Rs) = grb_vars
    # (rho, phi) = grb_exprs

    # Xs = [grb_vars_to_ndarray(X).astype(int) for X in Xs]
    # Zs = [grb_vars_to_ndarray(Z).astype(int) for Z in Zs]    
    # Rs = [grb_vars_to_ndarray(R) for R in Rs]
    # R_sup = grb_vars_to_ndarray(R_sup)

    # R_cat = np.concatenate(Rs, axis=-1)
    # Z_cat = np.concatenate(Zs, axis=-1)
    # print(R_cat.shape)
    # print(Z_cat.shape)
    # R_max = get_R_max(R_cat, Z_cat)
    # print(np.sum((R_max - R_sup) ** 2))
    # print(R_sup)

    # print("rho:", rho.getValue())
    # print("phi:", phi.getValue())