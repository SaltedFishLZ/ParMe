# experiment script for MPDL benchmark
import os
import sys
import time
import pickle
import logging
import calendar
import argparse
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


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='ParMe experiment parser.'
    )
    parser.add_argument('--t', '-t',
                        type=int, default=1,
                        help='number of templates')
    parser.add_argument('--theta', metavar='[0, 1]',
                        type=float, default=0.0,
                        help='weight for the cut size objective')
    parser.add_argument('--max-size',
                        type=float, required=True,
                        help='max size of each subgraph')
    parser.add_argument('--min-size',
                        type=float, required=True,
                        help='min size of each subgraph')
    parser.add_argument('--w0', metavar='%f', nargs='+',
                        type=float, default=None, 
                        help='node weights for each type of resource')
    parser.add_argument('--q', '-q', metavar='%f', nargs='+',
                        type=float, default=None, 
                        help='quantity weight for each G')
    parser.add_argument('--time', metavar='TIME',
                        type=float, default=5,
                        help='run time limit (min) for each optimization')
    parser.add_argument('--rho-star', metavar='rho*',
                        type=float, default=1.,
                        help='precalculated rho* used to scale rho')
    parser.add_argument('--phi-star', metavar='phi*',
                        type=float, default=1.,
                        help='precalculated phi* used to scale phi')
    return parser


def get_parameters(args, Gs, module_indices, echo=True):
    """parse ParMe parameters and make sanity check
    """
    n = len(Gs); r = len(module_indices)
    
    t = args.t
    assert t > 0, ValueError()

    theta = args.theta
    assert theta >= 0.0,  ValueError()
    assert theta <= 1.0, ValueError()
    
    if args.w0 is None:
        w0 = np.ones(r)
    else:
        w0 = np.asarray(args.w0)
        assert len(w0) == r, ValueError()
    
    if args.q is None:
        q = np.ones(n)
    else:
        q = np.asarray(args.q)
        assert len(q) == n, ValueError()
    
    max_size = args.max_size
    min_size = args.min_size
    assert max_size > min_size, ValueError()

    time = args.time
    assert time > 0., ValueError()

    rho_star = args.rho_star
    assert rho_star > 1e-20, ValueError()

    phi_star = args.phi_star
    assert phi_star > 1e-20, ValueError()

    if echo:
        print('[n]', n); print('[r]', r)
        print('[module/resource index]', module_indices)
        print('[t]', t)
        print('[theta]', theta)
        print('[w0]', w0)
        print('[q]', q)
        print('[max size]', max_size)
        print('[min size]', min_size)
        print('[time]', time)
        print('[rho_star]', rho_star)
        print('[phi_star]', phi_star)

    return (t, theta, w0, q, max_size, min_size, time, rho_star, phi_star)


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
    plt.clf(); plt.cla(); plt.close()


def plot_output(G: nx.Graph,
                X: np.ndarray,
                name: str):
    """
    :param G:
    :param X: $l \times m$
    """
    l = len(G.nodes)
    l_, m = X.shape
    assert l_ == l, ValueError('Shape mismatch')

    sgids = indicator_to_assignment(X, axis=1)

    # print('=' * 64)
    # print('# nodes: ', l)
    # print('# subgraphs:', m)
    # print('cluster id for nodes in ', name)
    # print(sgids)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G)
    
    nx.draw_networkx(G, ax=ax, pos=pos,
                     node_color=sgids,
                     cmap=plt.cm.magma, 
                     with_labels=False)

    dir_path = os.path.join('figures', 'mpdl')
    os.makedirs(dir_path, exist_ok=True)
    png_path = os.path.join(dir_path, '{}.png'.format(name))
    plt.savefig(png_path)
    plt.clf(); plt.cla(); plt.close()


if __name__ == "__main__":
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(format=FORMAT, level=logging.INFO)
    np.set_printoptions(edgeitems=30, linewidth=200)

    Gs = load_graphs()
    module_indices = get_module_index(Gs)

    # parse arguments
    parser = get_argparser()
    args = parser.parse_args()

    # calculate parameters from args
    parameters = get_parameters(args, Gs, module_indices)
    (t, theta, w0, q, max_size, min_size, time, rho_star, phi_star) = parameters

    # update inputs for ParMe
    print('=' * 64)
    print('reading input graphs')
    Ls = []; R0s = []
    for name in Gs:
        G = Gs[name]
        L = nx.laplacian_matrix(G).tocoo()
        Ls.append(L)
        R0 = get_R0(G, module_indices)
        R0s.append(R0)
        print(name)
        plot_input(G, name)
    
    # build Gurobi model
    model, grb_vars, grb_exprs = get_parme_model(Ls=Ls, q=q, R0s=R0s,
                                                 w0=w0, t=t, theta=theta,
                                                 min_size=min_size,
                                                 max_size=max_size,
                                                 rho_star=rho_star,
                                                 phi_star=phi_star)
    # update model and Gurobi configuration
    model.update()
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)
    model.setParam('TimeLimit', time * 60)

    model.optimize()

    if (model.status == GRB.OPTIMAL or
        model.status == GRB.TIME_LIMIT or
        model.status == GRB.NODE_LIMIT or
        model.status == GRB.ITERATION_LIMIT or
        model.status == GRB.USER_OBJ_LIMIT):
        # get a solution
        if model.status == GRB.TIME_LIMIT:
            print("[GRB] reach time limit")
        if model.status == GRB.OPTIMAL:
            print("[GRB] get optimal")
        
        if model.SolCount > 1:

            (Xs, Zs, R_sup, Rs) = grb_vars
            (rho, phi) = grb_exprs

            print("rho_tilde:", rho.getValue())
            print("phi_tilde:", phi.getValue())

            np_Xs = [grb_vars_to_ndarray(X, dtype=int) for X in Xs]
            np_Zs = [grb_vars_to_ndarray(Z, dtype=int) for Z in Zs]    
            np_Rs = [grb_vars_to_ndarray(R) for R in Rs]
            np_R_sup = grb_vars_to_ndarray(R_sup)

            # generate output figures
            for i, name in enumerate(Gs):
                # print(i, name)
                G = Gs[name]
                np_X = np_Xs[i]
                fig_name = "{}:t={:02d}:theta={:.03f}".format(
                    name, t, theta
                )
                plot_output(G, np_X, fig_name)

            # # check the optimality of R_sup
            # R_cat = np.concatenate(Rs, axis=-1)
            # Z_cat = np.concatenate(Zs, axis=-1)
            # print(R_cat.shape)
            # print(Z_cat.shape)
            # R_max = get_R_max(R_cat, Z_cat)
            # print(np.sum((R_max - R_sup) ** 2))
            # print(R_sup)

            exit(0)
        else:
            print('[GRB] no feasible solution found')
    elif (model.status == GRB.INFEASIBLE):
        print('[GRB] proved to be infeasible')
    elif (model.status == GRB.UNBOUNDED):
        print('[GRB] proved to be unbounded')
    else:
        print('[GRB] return code ', model.status)