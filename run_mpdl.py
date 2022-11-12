# experiment script for MPDL benchmark
import os
import sys
import time
import json
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

import utils
# from utils import *
from parme import get_parme_model

# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.{}.log".format(timestamp)


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
                name: str) -> None:
    """
    :param G:
    :param X: $l \times m$
    """
    l = len(G.nodes)
    l_, m = X.shape
    assert l_ == l, ValueError('Shape mismatch')

    sgids = utils.onehot_to_index(X, axis=1)

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
                     with_labels=True)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'output.{}.png'.format(name)))
    plt.savefig(os.path.join(out_dir, 'output.{}.pdf'.format(name)))

    plt.clf(); plt.cla(); plt.close()


if __name__ == "__main__":
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(format=FORMAT, level=logging.INFO)
    np.set_printoptions(edgeitems=30, linewidth=200)

    Gs = load_graphs()
    module_indices = get_module_index(Gs)

    # parse arguments
    parser = utils.get_argparser()
    args = parser.parse_args()

    # set gurobi log
    # global log_file
    log_file = os.path.join(args.output, log_file)

    # calculate parameters from args
    parameters = utils.get_parameters(args, n=len(Gs), r=len(module_indices))
    config = parameters

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
        plot_input(G=G, out_dir=args.output, name=name)
    print('=' * 64)
    
    # build Gurobi model
    model, grb_vars, grb_exprs = get_parme_model(Ls=Ls, R0s=R0s,
                                                 q=config['q'],
                                                 w0=config['w0'],
                                                 t=config['t'],
                                                 theta=config['theta'],
                                                 min_size=config['min_size'],
                                                 max_size=config['max_size'],
                                                 rho_star=config['rho_star'],
                                                 phi_star=config['phi_star'])
    # update model and Gurobi configuration
    model.update()
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)
    model.setParam('TimeLimit', config['time'] * 60)

    model.optimize()

    # obtain GRB status
    grbret = {'status' : model.status,}
    
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
        
        grbret['solcount'] = model.SolCount

        if model.SolCount > 0:
            grbret['bestobj'] = model.getObjective().getValue()

            (Xs, Zs, R_sup, Rs) = grb_vars
            (rho, phi) = grb_exprs

            grbret['rho_tilde'] = rho.getValue()
            grbret['phi_tilde'] = phi.getValue()
            grbret['rho'] = grbret['rho_tilde'] * config['rho_star']
            grbret['phi'] = grbret['phi_tilde'] * config['phi_star']

            Xs = [utils.grb_vars_to_ndarray(X, dtype=int) for X in Xs]
            Zs = [utils.grb_vars_to_ndarray(Z, dtype=int) for Z in Zs]
            Rs = [utils.grb_vars_to_ndarray(R, dtype=int) for R in Rs]
            R_sup = utils.grb_vars_to_ndarray(R_sup, dtype=int)

            # sort X assignment, re-arrange subgraph id
            for i in range(len(Xs)):
                # X: l \times m, assignee is subgraph index
                Xs[i] = Xs[i].T
                Xs[i], index = utils.sorted_assignment(Xs[i], axis=0, with_index=True)
                Xs[i] = Xs[i].T
                Zs[i] = Zs[i][:, index]
                Rs[i] = Rs[i][:, index]

            # generate output figures
            for i, name in enumerate(Gs):
                # print(i, name)
                G = Gs[name]
                X = Xs[i]
                plot_output(G=G, X=X, out_dir=args.output, name=name)
            
            # dump np results to pickle files
            utils.dump_pickle_results(out_dir=args.output,
                                      Xs=Xs, Zs=Zs, Rs=Rs, R_sup=R_sup)
            # dump np results to text files
            utils.dump_text_results(out_dir=args.output,
                                      Xs=Xs, Zs=Zs, Rs=Rs, R_sup=R_sup)

        else:
            print('[GRB] no feasible solution found')
    elif (model.status == GRB.INFEASIBLE):
        print('[GRB] proved to be infeasible')
    elif (model.status == GRB.UNBOUNDED):
        print('[GRB] proved to be unbounded')
    else:
        print('[GRB] return code ', model.status)

    # optimization status
    print('=' * 64)
    print("\n".join("{}: {}".format(k + ' ' * (32 - len(k)), v)
                    for k, v in grbret.items()))

    # dump config to json
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'grbret.json'), 'w') as fp:
        json.dump(grbret, fp, indent=4)