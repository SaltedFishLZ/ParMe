import os
import sys
import pickle


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
import matplotlib as mpl
import matplotlib.pyplot as plt 

import EleNetX.mpdl as mpdl

import utils
import visualize

# tmp hack
import run_mpdl

import visualize


if __name__ == "__main__":
    

    Gs = run_mpdl.load_graphs()
    module_indices = run_mpdl.get_module_index(Gs)

    # load results
    out_dir = 'exp_dir'

    (Xs, Zs, Rs, R_sup) = utils.load_pickle_results(out_dir)

    for i, name in enumerate(Gs.keys()):
        G = Gs[name]
        X = Xs[i]

        visualize.plot_output(G, X, out_dir, name)

