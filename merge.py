import os
import time
import pickle
import logging
import calendar
from typing import Union, List

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import grb_vars_to_ndarray, grb_vars_shape, \
                  sorted_assignment


tensor = Union[np.ndarray, gp.tupledict]

# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.merge.{}.log".format(timestamp)


def get_merge_constraints(R_sup: gp.tupledict,
                          Z: gp.tupledict,
                          R: np.ndarray) -> dict:
    """
    :param Z: shape $t \times m$, t templates, m subgraphs
    :param R_sup: shape $r \times t$, r resources, t templates
    :param R: shape $r \times m$
    """
    t, m = grb_vars_shape(Z)
    r, t_ = grb_vars_shape(R_sup); assert t == t_, ValueError()
    r_, m_ = R.shape; assert r == r_, ValueError(); assert m == m_, ValueError();

    # (1) unique template assignment
    # each subgraph will be assign to EXACTLY merged design
    unique_assign = (Z.sum('*', j) == 1 for j in range(m))
    # (2) resource upper bound:
    # each template must have SUFFICIENT resources to cover every
    # subgraph for all types of resources
    resource_ub = (R_sup[i, j] >= R[i, k] * Z[j, k]
                   for i in range(r) # for each type of resource
                   for j in range(t)
                   for k in range(m))

    return {
        "Merge:unique-assignment"       :   unique_assign  ,
        "Merge:resource-upper-bound"    :   resource_ub    ,
    }


def get_merge_objective(R_sup: gp.tupledict,
                        Z: gp.tupledict,
                        q: np.ndarray,
                        w0: np.ndarray) -> gp.QuadExpr:
    """
    :param R_sup: resource upper bound $\overline{R}$,
                  shape $r \times t$, r resources, t templates
    :param Z: template assignment, 
              shape $t \times m$, t templates, m subgraphs
    :param q: quantity/volume for each subgraph, shape $m$
    :param w0: weight for each type of resource, shape $r$
    :return: a Gurobi expression
    """
    t, m = grb_vars_shape(Z)
    r, t_ = grb_vars_shape(R_sup); assert t == t_, ValueError()
    (r_, ) = w0.shape; assert r == r_, ValueError()
    (m_, ) = q.shape; assert m == m_, ValueError()

    # s: area for each template, a list
    S = [gp.quicksum(w0[i] * R_sup[i, j] for i in range(r)) for j in range(t)]

    # RE cost for each template, a list
    # TODO: plug in Seed's model
    Phi = S

    # q_sup: aggregated product volumes for each template
    q_sup = [ gp.quicksum(Z[k, j] * q[j] for j in range(m))
          for k in range(t) ] # template k in [t]

    # total RE cost for an input graph
    cost = gp.quicksum(Phi[k] * q_sup[k] for k in range(t))

    return cost


def get_merge_model(R: tensor,
                    t: int,
                    q: np.ndarray,
                    w0: np.ndarray):
    """
    :param R: shape $r \times m$
    :param t: number of templates
    :param q: q vector, shape m
    :param w0: node weight of each resource, shape r
    """
    if isinstance(R, np.ndarray):
        r, m = R.shape
        # print("R = ")
        # print(R)
    elif isinstance(R, gp.tupledict):
        r, m = grb_vars_shape(R)
    (m_, ) = q.shape; assert m_ == m, ValueError()
    (r_, ) = w0.shape; assert r_ == r, ValueError()

    model = gp.Model("bilinear graph merge")
    
    # set variables
    Z = model.addVars(t, m, name="Z", vtype=GRB.BINARY)
    R_sup = model.addVars(r, t, name="R_sup")

    # set constraints
    merge_constrs = get_merge_constraints(R_sup=R_sup,
                                          Z=Z,
                                          R=R)
    for constrs in merge_constrs:
        model.addConstrs(merge_constrs[constrs], name=constrs)

    # set objective
    cost = 0.0
    cost += get_merge_objective(R_sup=R_sup,
                                Z=Z,
                                q=q,
                                w0=w0)
    model.setObjective(cost, GRB.MINIMIZE)

    # add Gurobi configuration and update model 
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)
    model.setParam('TimeLimit', 20 * 60)
    model.update()

    return model, Z, R_sup


def get_merge_nG_model(Rs: List[tensor],
                       t: int,
                       qs: List[np.ndarray],
                       w0: np.ndarray):
    """
    :param Rs: 
    :param t: number of templates
    :param qs: 
    :param w0: node weight of each resource, shape r
    """
    model = gp.Model("bilinear graph merge with n groups")

    n = len(Rs); assert n == len(qs), ValueError()
    (r,) = w0.shape

    # init objective and variables
    cost = 0.0
    Zs = []
    R_sup = model.addVars(r, t, name="R_sup") # R_sup is shared

    for i in range(n):
        R = Rs[i]; q = qs[i]

        if isinstance(R, np.ndarray):
            r, m = R.shape
            # print("R = ")
            # print(R)
        elif isinstance(R, gp.tupledict):
            r, m = grb_vars_shape(R)
        (m_, ) = q.shape; assert m_ == m, ValueError()
        (r_, ) = w0.shape; assert r_ == r, ValueError()

        # set variables
        Z = model.addVars(t, m, name="Z", vtype=GRB.BINARY)
        Zs.append(Z)

        # set constraints
        merge_constrs = get_merge_constraints(R_sup=R_sup,
                                              Z=Z,
                                              R=R)
        for constrs in merge_constrs:
            model.addConstrs(merge_constrs[constrs], name=constrs)

        # set objective
        cost += get_merge_objective(R_sup=R_sup,
                                    Z=Z,
                                    q=q,
                                    w0=w0)

    model.setObjective(cost, GRB.MINIMIZE)

    # add Gurobi configuration and update model 
    model.setParam("LogFile", log_file)
    model.setParam("LogToConsole", 0)
    model.setParam('TimeLimit', 20 * 60)
    model.update()

    return model, Zs, R_sup


if __name__ == "__main__":
    # read input
    fpath = os.path.join("data", "random", "random.003.pkl")
    with open(fpath, "rb") as fin:
        fdict = pickle.load(fin)
    
    # meta data
    r = fdict["r"]; m = fdict["l"]; t = fdict["m"]
    print("r =", r, "; m =", m, "; t =", t)

    R = fdict["R"]; w0 = fdict["c"]; q = fdict["v"]
    print(q)

    # hack
    t = 5
    R = np.concatenate([R, R], axis=-1)
    q = np.concatenate([q, q], axis=-1)

    model, Z, R_sup = get_merge_model(R=R, q=q, t=t, w0=w0)

    # Optimize
    model.optimize()

    Z0 = grb_vars_to_ndarray(Z).astype(int)
    R_sup0 = grb_vars_to_ndarray(R_sup)
    print(Z0)
    print(R_sup0)

    # sort Z0
    print("=" * 40); print("sorted"); print("=" * 40)
    Z0, index = sorted_assignment(Z0, axis=0, with_index=True)
    R_sup0 = R_sup0[:, index]
    print(Z0)
    print(R_sup0)

    Rs = [R[:, 0 : (m // 2)], R[:, (m // 2) : ]]
    qs = [q[0 : (m // 2)], q[(m // 2) : ]]
    model, Zs, R_sup = get_merge_nG_model(Rs=Rs, qs=qs, t=t, w0=w0)

    model.optimize()
    Zs = [grb_vars_to_ndarray(Z).astype(int) for Z in Zs]
    Z1 = np.concatenate(Zs, axis=1)
    R_sup1 = grb_vars_to_ndarray(R_sup)
    print(Z1)
    print(R_sup1)

    print("=" * 40); print("sorted"); print("=" * 40)
    Z1, index = sorted_assignment(Z1, axis=0, with_index=True)
    R_sup1 = R_sup1[:, index]
    print(Z1)
    print(R_sup1)

    assert (Z0 == Z1).all(), ValueError()
    square_error = np.sum(np.square(R_sup0 - R_sup1))
    assert square_error < 1e-10, ValueError()
