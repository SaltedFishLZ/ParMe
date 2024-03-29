import os
import time
import pickle
import logging
import calendar
from typing import Union, List

import numpy as np
import scipy.sparse as sp
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import matplotlib as mpl
import matplotlib.pyplot as plt 

from EleNetX.mcnc import yal_to_nx, PD91_BASE_DIR
import EleNetX.mpdl as mpdl
from EleNetX.utils import obj_attr_cat_to_int
from EleNetX.visualize import plot_ele_nx

from utils import *
from partition import get_partition_constraints, \
                      get_partition_objective

from merge import get_merge_constraints, \
                  get_merge_objective


tensor = Union[np.ndarray, gp.tupledict]

# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.ParMe.{}.log".format(timestamp)


def get_parme_model(
    Ls: List[np.ndarray],
    q: np.ndarray,
    R0s: List[np.ndarray],
    w0: np.ndarray,
    t: int,
    max_size,
    min_size,
    theta,
    alpha: int = 10,
    d0: float = 0.09 * 1e-2,
    rho_star: float = 1.0,
    phi_star: float = 1.0
):
    """
    :param alpha: critical levels
    :param: d0 defects per mm^2
    """
    # sanity check
    n = len(Ls); assert n == len(q)
    (r,) = w0.shape
    
    model = gp.Model("ParMe")

    # init partition objective and variables
    rho = 0.0; Xs = []
    # connection variables
    Rs = []
    # init merge objectives and variables
    phi = 0.0; Zs = []
    R_sup = model.addVars(r, t, name="R_sup", lb=0.0)

    # -------------------------------- #
    #         Partition Part           #
    # -------------------------------- #
    # iter over each input G and update variables, constrs, and obj
    for i in range(n):
        L = Ls[i]; R0 = R0s[i]
        l, l_ = L.shape; assert l == l_, ValueError()
        # connection equation:
        # w = w0^T R0 (w is a constant for each G)
        w = w0.T @ R0
        # print('w =', w)
        # get m subgraphs
        print('design size =', w.sum())
        m = int(np.floor(w.sum() / min_size))
        m = min(l, m)
        print("m = ", m)
        # set variables
        X = model.addVars(l, m, vtype=GRB.BINARY)
        Xs.append(X)
        # set constraints
        # the merge will enlarge the size,
        # we allow each original subgraph to take 70% of the space of the final chiplet
        partition_constrs = get_partition_constraints(X=X, w=w, 
                                                      max_size=max_size * 0.7,
                                                      min_size=min_size * 0.7)
        for constrs in partition_constrs:
            model.addConstrs(partition_constrs[constrs], name=constrs)
        # the cut size of an input G^i
        rho_i = get_partition_objective(X, L)
        # quantity-weighted
        rho += q[i] * rho_i

    # -------------------------------- #
    #    Template Auxilary Variables   #
    # -------------------------------- #
    # Auxilary variables shared across all input graphs
    # s: area (mm^2) for each template
    # 800 mm^2 area upper bound
    # NOTE: change it to a parameter
    s = model.addVars(t, name="template area", lb=0.0, ub=8e2)
    model.addConstrs(s[k] >= gp.quicksum(w0[i] * R_sup[i, k] for i in range(r))
                        for k in range(t))
    # phi_template: template cost
    phi_template = model.addVars(t, name="template cost")
    # calculate the RE cost using
    # 3rd order expansion
    p0 = 0.0; p1 = 1.0; p2 = d0
    p3 = (alpha - 1) / (2 * alpha) * (d0 ** 2)
    for k in range(t):
        constr = model.addGenConstrPoly(xvar=s[k],
                                        yvar=phi_template[k],
                                        p=[p3, p2, p1, p0],
                                        name='T_{} RE cost'.format(k))
        model.update()
        constr.setAttr('FuncPieceError', 1e-4)

    # -------------------------------- #
    #           Merge Part             #
    # -------------------------------- #
    # iter over each input G and update variables, constrs, and obj
    for i in range(n):
        X = Xs[i]; R0 = R0s[i]
        l, m = grb_vars_shape(X)
        r_, l_ = R0.shape
        assert l_ == l, ValueError()
        assert r_ == r, ValueError()
        # connection equation as constraints:
        # R = R0 X
        R = model.addVars(r, m, name="R", lb=0.0)
        Rs.append(R)
        model.addConstrs(R[i, j] == gp.quicksum(R0[i, k] * X[k, j] for k in range(l))
                         for i in range(r)
                         for j in range(m))
        # assignment variables
        Z = model.addVars(t, m, name="Z", vtype=GRB.BINARY)
        Zs.append(Z)

        # q_sup: aggregated product volumes for each template
        # NOTE: the scope is limited to input G^i
        q_sup = [ gp.quicksum(Z[k, j] for j in range(m))
                  for k in range(t) ] # template k in [t]

        # set constraints
        merge_constrs = get_merge_constraints(R_sup=R_sup, Z=Z, R=R)
        for constrs in merge_constrs:
            model.addConstrs(merge_constrs[constrs], name=constrs)
        
        # get the RE costs of all subgraphs in G^i
        phi_i = gp.quicksum(phi_template[k] * q_sup[k] for k in range(t))

        # weighted by quantity
        phi += q[i] * phi_i

    # normalize 2 objectives
    rho_tilde = rho / rho_star
    phi_tilde = phi / phi_star

    # set Gurobi objective
    objective = theta * rho_tilde + (1 - theta) * phi_tilde
    model.setObjective(objective)
    model.update()

    return model, (Xs, Zs, R_sup, Rs), (rho_tilde, phi_tilde)


if __name__ == "__main__":
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # graphs = []; gnames = []
    # for neural_net in mpdl.MPDL_BENCHMARKS.keys():
    #     num_cfg = mpdl.MPDL_BENCHMARKS[neural_net]['num_cfg']
    #     # enumerate configs
    #     for i in range(num_cfg):
    #         gpath = mpdl.compose_graph_path(neural_net, i)
    #         print(gpath)
    #         G = nx.read_gpickle(gpath)

    #         mpdl.assign_modules(G)

    #         graphs.append(G)
    #         gnames.append(neural_net + ':' + str(i))
    
    # n = len(Gs)
    # q = np.ones(n)

    # Ls = []; R0s = 0
    # for G in graphs:
    #     L = nx.laplacian_matrix(G)
    #     Ls.append(L)


    category = "stdcell"
    benchmark = "fract"        
    yal_path = os.path.join(PD91_BASE_DIR,
                            category,
                            benchmark + ".yal")

    G = yal_to_nx(yal_path)
    # hack
    sub_nodes = list(G.nodes)[0:50]
    G = G.subgraph(sub_nodes)
    
    module_indices = obj_attr_cat_to_int(G.nodes(), "modulename")
    print(module_indices)

    l = len(G.nodes)
    print("G has", l, "nodes")
    r = len(module_indices)

    # obtain coef data
    L = nx.laplacian_matrix(G)
    L = L.tocoo()

    R0 = np.zeros((r, l))
    for j, v in enumerate(G.nodes):
        k = module_indices[G.nodes[v]["modulename"]]
        R0[k, j] = 1
    
    # print(R0)
    # print(l)

    n = 2
    Ls = [L, ] * n
    R0s = [R0, ] * n
    q = np.asarray([1, 2])
    w0 = np.ones(r)

    t = 2; theta = 0.2
    min_size = 8
    max_size = 16


    model, grb_vars, grb_exprs = get_parme_model(Ls=Ls, q=q, R0s=R0s, w0=w0, t=t,
                                                 min_size=min_size, max_size=max_size,
                                                 theta=theta)
    # update model and Gurobi configuration
    model.update()
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)
    model.setParam('TimeLimit', 1 * 60)

    model.optimize()

    (Xs, Zs, R_sup, Rs) = grb_vars
    (rho, phi) = grb_exprs

    Xs = [grb_vars_to_ndarray(X, dtype=int) for X in Xs]
    Zs = [grb_vars_to_ndarray(Z, dtype=int) for Z in Zs]    
    Rs = [grb_vars_to_ndarray(R) for R in Rs]
    R_sup = grb_vars_to_ndarray(R_sup)

    R_cat = np.concatenate(Rs, axis=-1)
    Z_cat = np.concatenate(Zs, axis=-1)
    print(R_cat.shape)
    print(Z_cat.shape)
    R_max = get_R_max(R_cat, Z_cat)
    print(np.sum((R_max - R_sup) ** 2))
    print(R_sup)

    print("rho:", rho.getValue())
    print("phi:", phi.getValue())