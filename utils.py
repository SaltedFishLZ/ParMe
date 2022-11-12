# __all__ = [
#     "grb_vars_shape",
#     "grb_vars_to_ndarray",
#     "onehot_to_index",
#     "assignment_to_cluster",
#     "indicator_to_cluster",
#     "sorted_assignment",
#     "get_R_max",
#     'dump_pickle_results',
#     'load_pickle_results'
# ]

import os
import sys
import time
import copy
import json
import pickle
import argparse

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


def onehot_to_index(onehot: np.ndarray,
                 axis: int = 1) -> np.ndarray:
    """Convert one-hot assignment tensor to a tensor of assignee ids
    for example, we will convert [[0 0 0 1 0],
                                  [0 1 0 0 0]]
    to [3, 1]
    :param indicator: an one-hot assignment
    :param axis: assignee dimension
    :return index: assignee id tensor
    """
    index = np.argmax(onehot, axis=axis)
    return index


def assignment_to_catalog(assignment: np.ndarray) -> dict:
    """Convert an assignment id tensor to a catalog dict
    representing the content of each assignee
    key: assignee
    value: a list of tasks
    """
    # only support 2d assignment: 1d tasks
    assert len(assignement.shape) == 2, NotImplementedError()
    # a pointer dict contains the indices/pointers of
    # tasks/jobs/content for each assignee
    catalog = dict()
    # update catalog
    for task, assignee in enumerate(list(assignment)):
        if assignee in catalog:
            catalog[assignee].append(task)
        else:
            cluster[assignee] = [task, ]
    return catalog


def indicator_to_string(vec1: np.ndarray) -> str:
    """get the string representation from the 0-1 indicator
    of an assignee
    """
    assert len(vec1.shape) == 1, ValueError()
    s = ''.join(map(str, vec1))
    return s


def string_to_indicator(s: str) -> np.ndarray:
    """get the the 0-1 indicator of an assignee from the string
    """
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
            s = indicator_to_string(a[i, :])
            strings.append(s)
    
        sorted_strings = sorted(strings)
        index = np.argsort(strings)

        a_sorted = [string_to_indicator(s) for s in sorted_strings]
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


# ---------------------------------------------------------------- #
#                   parser and configs                             #
# ---------------------------------------------------------------- #

def get_argparser(name='experiment parser') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=name
    )
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='output directory')
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


def get_parameters(args, n, r, echo=True):
    """obtain ParMe configs from args and make basic sanity checks
    """
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

    config = {
        'output' : args.output,
        't' : t,
        'theta' : theta,
        'w0' : w0,
        'q' : q,
        'max_size' : max_size,
        'min_size' : min_size,
        'time' : time,
        'rho_star' : rho_star,
        'phi_star' : phi_star
    }

    if echo:
        print("\n".join("{}: {}".format(k + ' ' * (32 - len(k)), v)
                        for k, v in config.items()))

    # dump config to json
    jsonable_config = copy.deepcopy(config)
    for key in jsonable_config:
        val = jsonable_config[key]
        if isinstance(val, np.ndarray):
            assert len(val.shape) == 1, NotImplementedError()
            # NOTE: please use dict[key] to modify
            # the content of a dictionary record
            jsonable_config[key] = val.tolist()

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'config.json'), 'w') as fp:
        json.dump(jsonable_config, fp, indent=4)

    return config


