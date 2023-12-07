import os
import sys
import pickle
import argparse

if sys.version <= "3.7":
    try:
        from collections.abc import OrderedDict
    except ImportError:
        from collections import OrderedDict
else:
    OrderedDict = dict

sys.path.append(os.getcwd())

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt 

import EleNetX.mpdl as mpdl

import utils
import visualize

# tmp hack
import run_mpdl

import visualize


parser = argparse.ArgumentParser(
    description='convert results to JSON files'
)
parser.add_argument('--output', '-o', type=str, required=True,
                    help='output directory')


if __name__ == "__main__":

    args = parser.parse_args()

    Gs = run_mpdl.load_graphs()
    module_indices = run_mpdl.get_module_index(Gs)

    out_dir = args.output


    # load results
    (Xs, Zs, Rs, R_sup) = utils.load_pickle_results(out_dir)

    rho = 0.0
    for i, name in enumerate(Gs):
        G = Gs[name]; X = Xs[i]; Z = Zs[i]
        print(name)
        L = nx.laplacian_matrix(G=G)
        
        l, l_ = L.shape; assert l_ == l, ValueError()
        l_, m = X.shape; assert l_ == l, ValueError()

        rho_i = 0.0
        for j in range(m):
            x_j = X[:, j]
            rho_i += x_j.T @ L @ x_j
        rho_i = 0.5 * rho_i

        print('cut size (GiB/s)', rho_i / 1e9)

        A = nx.adjacency_matrix(G=G)
        print(A.sum() / 1e9 / 2)
        print(G.size(weight='weight') / 1e9)

        rho += rho_i
    

    print('rho by parme is', rho)