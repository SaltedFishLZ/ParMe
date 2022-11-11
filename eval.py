from typing import List

import numpy as np
import scipy.sparse as sp
import networkx as nx

from utils import indicator_to_assignment


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


def get_R_max_nG(Rs: List[np.ndarray],
                 Zs: List[np.ndarray]) -> np.ndarray:
    # sanity check
    assert len(Rs) == len(Zs), ValueError()
    n = len(Rs)
    for i in range(n):
        assert isinstance(Rs[i], np.ndarray), ValueError()
        assert isinstance(Zs[i], np.ndarray), ValueError()
        r, m = Rs[i].shape; t, m_ = Zs[i].shape
        assert m_ == m, ValueError("Size mismatch")
    # concat all sub-graphs in a single canvas/disconnected graph
    R_cat = np.concatenate(Rs, axis=-1)
    Z_cat = np.concatenate(Zs, axis=-1)

    R_max = get_R_max(R=R_cat, Z=Z_cat)

    return R_max


def get_cut_size_np(L, X):
    assert isinstance(X, np.ndarray), ValueError()
    l, l_ = L.shape; assert l == l_, ValueError()
    l_, m = X.shape; assert l == l_, ValueError()
    cut_size = 0.0
    for j in range(m):
        x_j = X[:, j]
        # cut size between subgraph j and the remaining part
        cut_j = x_j.T @ L @ x_j
        cut_size += cut_j
    return cut_size / 2.0


def get_cut_size_nx(G: nx.Graph, X: np.ndarray) -> float:
    """
    """
    # sanity check
    assert isinstance(G, nx.Graph), TypeError()
    assert isinstance(X, np.ndarray), TypeError()
    l = len(G.nodes); l_, m = X.shape; assert l == l_, ValueError()
    # get node indices for each subgraph
    sgids = indicator_to_assignment(X, axis=1)
    sg_v_map = dict()
    for j in range(m):
        sg_v_map[j] = []
    for k in range(l):
        j = sgids[k]
        sg_v_map[j].append(k)
    # accumulate cut size
    cut_size = 0.0
    for j1 in range(m):
        for j2 in range(j1 + 1, m):
            nids1 = [(k + 1) for k in sg_v_map[j1]]
            nids2 = [(k + 1) for k in sg_v_map[j2]]
            cut_j1j2 = nx.cut_size(G, nids1, nids2, weight='weight')
            cut_size += cut_j1j2

    return cut_size


def get_subG_size(X, w):
    return X.T @ w


def get_rho(Xs, Ls, q):
    rho = 0.0
    n = len(Xs); n_ = len(Ls)
    assert n == n_, ValueError()
    for i in range(n):
        L = Ls[i]; X = Xs[i]
        rho_i =  get_cut_size_np(X=X, L=L.todense())
        print(rho_i)
        rho += q[i] * rho_i
    return rho


def get_rho_nx(Xs, Gs, q):
    n = len(Xs); n_ = len(Gs)
    assert n == n_, ValueError()

    gnames = list(Gs.keys())
    rho = 0.0
    for i in range(n):
        G = Gs[gnames[i]]; X = Xs[i]
        rho_i =  get_cut_size_nx(X=X, G=G)
        print(rho_i)
        rho += q[i] * rho_i
    return rho



def get_T_size(R_sup, w0):
    return w0.T @ R_sup


def get_T_quantity(Z, q):
    return q.T @ Z


def get_T_cost(R, q, w0):
    pass