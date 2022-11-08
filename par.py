import os
import time
import logging
import calendar

import numpy as np
import scipy.sparse as sp
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from utils import grb_vars_shape, grb_vars_to_ndarray
from EleNetX.mpdl import *
from EleNetX.visualize import plot_ele_nx


def get_par_constraints(X:gp.tupledict,
                        # w:np.ndarray,
                        max_size,
                        min_size) -> dict:
    """ Get partition constraints for each individual graph
    :param X: partition 0-1 variable, shape $l \times m$,
    l nodes in total, m subgraphs
    :param w: node weight, constant, shape l
    :param max_size:
    :param min_size:
    """
    l, m = grb_vars_shape(X)
    # (1) unique cluster assignment
    unique_assign = (X.sum(k, '*') == 1 for k in range(l))
    # (2) max size
    size_ub = (X.sum('*', j) <= max_size for j in range(m))
    # (3) min size
    size_lb = (X.sum('*', j) >= min_size for j in range(m))

    return {
        "Par:unique-subgraph-assignment"    :   unique_assign,
        "Par:cluster-size-upper-bound"      :   size_ub,
        "Par:cluster-size-lower-bound"      :   size_lb
    }


def get_par_objective(X:gp.tupledict,
                      L:sp.coo_matrix) -> gp.QuadExpr:
    """ Get partition objective for each individual graph
    """
    l, m = grb_vars_shape(X)
    # enumerate node k1, k2, and subgraph j
    cut_size = gp.quicksum(X[k1, j] * val * X[k2, j]
                           for (k1, k2, val) in zip(L.row, L.col, L.data)
                           for j in range(m))
    cut_size = cut_size * 0.5 # remove duplication
    return cut_size


def get_par_model(L, m=2, gamma=0.1):
    """
    """
    model = gp.Model("quadratic-programming graph partition")

    l, _ = L.shape
    max_size = np.ceil(l / m) * (1 + gamma)

    # set variables
    X = model.addVars(l, m, vtype=GRB.BINARY)

    # set constraints
    par_constrs = get_par_constraints(X, max_size, 0)
    for constrs in par_constrs:
        model.addConstrs(par_constrs[constrs], name=constrs)

    # set objective
    cut_size = get_par_objective(X, L)
    model.setObjective(cut_size)

    # Gurobi configuration
    current_GMT = time.gmtime()
    timestamp = calendar.timegm(current_GMT)
    log_file = "gurobi.par.{}.log".format(timestamp)
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)

    model.update()
    return model, X


if __name__ == "__main__":
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)


    graphs = []
    gnames = []

    neural_net = 'resnet50'
    gpath = compose_graph_path(neural_net, 0)
    print(gpath)

    G = nx.read_gpickle(gpath)
    L = nx.laplacian_matrix(G)
    L = L.tocoo()

    l, _ = L.shape
    m = 2
    print(l)

    model, X = get_par_model(L, m=4)
    model.optimize()

    X = grb_vars_to_ndarray(X).astype(int)
    print(X)
