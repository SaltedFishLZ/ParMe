__all__ = [
    "grb_vars_shape",
    "grb_vars_to_ndarray",
    "indicator_to_assignment",
    "assignment_to_cluster",
    "indicator_to_cluster"
]

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def grb_vars_shape(vars:gp.tupledict) -> tuple:
    """Get the shape of a multi-dimensional Gurobi Vars
    NOTE: vars should be a continuous dense array
    :param vars: a Gurobi multi-dimensional Vars
    :return: the shape of :vars:
    """
    coordinates = list(vars.keys())
    shape = list(coordinates[0])
    ndim = len(shape)
    for coord in coordinates:
        assert len(coord) == len(shape)
        for (i, x_i) in enumerate(coord):
            if x_i > shape[i]:
                shape[i] = x_i
    # append 1
    for i in range(ndim):
        shape[i] += 1

    return tuple(shape)


def grb_vars_to_ndarray(vars:gp.tupledict,
                        shape:tuple=None) -> np.ndarray:
    """Convert a multi-dimensional Gurobi Vars to a numpy ndarray
    :param model:
    :param vars:
    :param shape:
    :return:
    """
    # calculate array shape
    if shape is None:
        shape = grb_vars_shape(vars)

    # fill in values
    array = np.zeros(shape)
    for coord in vars:
        var = vars[coord]
        array[coord] = var.X

    return array


def indicator_to_assignment(indicator:np.ndarray) -> np.ndarray:
    """
    """
    assignment = np.argmax(indicator, axis=1)
    return assignment


def assignment_to_cluster(assignment:np.ndarray) -> dict:
    cluster = dict()
    for i, c in enumerate(list(assignment)):
        if c in cluster:
            cluster[c].append(i)
        else:
            cluster[c] = [i, ]
    return cluster


def indicator_to_cluster(indicator:np.ndarray) -> dict:
    assignment = indicator_to_assignment(indicator)
    cluster = assignment_to_cluster(assignment)
    return cluster