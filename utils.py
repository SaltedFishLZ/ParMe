__all__ = [
    "grb_vars_shape",
    "grb_vars_to_ndarray",
    "indicator_to_assignment",
    "assignment_to_cluster",
    "indicator_to_cluster",
    "sorted_assignment",
    "get_R_max",
    'dump_pickle_results',
    'load_pickle_results'
]

import os
import sys
import time
import json
import pickle

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


def grb_vars_to_ndarray(vars: gp.tupledict,
                        shape: tuple = None,
                        dtype: type = float) -> np.ndarray:
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
    array = np.zeros(shape, dtype=dtype)
    for coord in vars:
        var = vars[coord]
        val = var.X
        if dtype == int:
            array[coord] = np.rint(val)
        elif dtype == float:
            array[coord] = float(val)
        else:
            raise NotImplementedError()

    return array


def indicator_to_assignment(indicator: np.ndarray,
                            axis: int = 1) -> np.ndarray:
    """
    """
    assignment = np.argmax(indicator, axis=axis)
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


def assignment_to_string(a: np.ndarray) -> str:
    """get the string representation of each assignee
    """
    s = ''.join(map(str, a))
    return s


def string_to_assignment(s: str) -> np.ndarray:
    a = [*s]
    a = [int(c) for c in a]
    return np.asarray(a, dtype=int)


def sorted_assignment(a: np.ndarray,
                      axis: int,
                      with_index=False) -> np.ndarray:
    """Sort assignment matrix as strings
    :param axis: which axis is the assignee
    """
    assert len(a.shape) == 2, NotImplementedError()
    assert axis == 0, NotImplementedError()

    strings = []
    if (axis == 0):
        for i in range(a.shape[axis]):
            s = assignment_to_string(a[i, :])
            strings.append(s)
    
        sorted_strings = sorted(strings)
        index = np.argsort(strings)

        a_sorted = [string_to_assignment(s) for s in sorted_strings]
        a_sorted = np.stack(a_sorted, axis=axis)
    else:
        raise NotImplementedError()

    if with_index:
        return (a_sorted, index)
    else:
        return a_sorted

def get_R_max(R: np.ndarray,
              Z: np.ndarray):
    """
    :paran R: size $r \times m$
    :param Z: size $t \times m$
    :return: R_max ($t \times m$), $\overline{R}$ by using max function
    """
    assert isinstance(R, np.ndarray), NotImplementedError()
    assert isinstance(Z, np.ndarray), NotImplementedError()
    r, m = R.shape; t, m_ = Z.shape
    assert m_ == m, ValueError("Size mismatch")
    # use einsum without reduction
    R_max = np.max(np.einsum('ij,kj->ikj', R, Z), axis=-1)
    return R_max


def dump_pickle_results(out_dir: str, Xs, Zs, Rs, R_sup):
    with open(os.path.join(out_dir, 'Xs.pkl'), 'wb') as file:
        pickle.dump(Xs, file)
    with open(os.path.join(out_dir, 'Zs.pkl'), 'wb') as file:
        pickle.dump(Zs, file)
    with open(os.path.join(out_dir, 'Rs.pkl'), 'wb') as file:
        pickle.dump(Rs, file)
    with open(os.path.join(out_dir, 'R_sup.pkl'), 'wb') as file:
        pickle.dump(R_sup, file)


def load_pickle_results(out_dir: str):
    with open(os.path.join(out_dir, 'Xs.pkl'), 'rb') as file:
        Xs = pickle.load(file)
    with open(os.path.join(out_dir, 'Zs.pkl'), 'rb') as file:
        Zs = pickle.load(file)
    with open(os.path.join(out_dir, 'Rs.pkl'), 'rb') as file:
        Rs = pickle.load(file)
    with open(os.path.join(out_dir, 'R_sup.pkl'), 'rb') as file:
        R_sup = pickle.load(file)
    
    return (Xs, Zs, Rs, R_sup)


def dump_text_results(out_dir: str, Xs, Zs, Rs, R_sup):
    """
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, X in enumerate(Xs):
        np.savetxt(os.path.join(out_dir, 'X-{}.txt'.format(i)),
                   X, fmt='%d')
    for i, Z in enumerate(Zs):
        np.savetxt(os.path.join(out_dir, 'Z-{}.txt'.format(i)),
                   Z, fmt='%d')
    for i, R in enumerate(Rs):
        np.savetxt(os.path.join(out_dir, 'R-{}.txt'.format(i)) ,
                   R, fmt='%d')
    np.savetxt(os.path.join(out_dir, 'R_sup.txt'),
               R_sup, fmt='%d')


def dump_json_results():
    pass