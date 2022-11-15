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

    # dump results
    utils.dump_json_results(Xs=Xs, Zs=Zs, Rs=Rs, R_sup=R_sup,
                            Gs=Gs, out_dir=out_dir)

    # utils.dump_text_results(Xs=Xs, Zs=Zs, Rs=Rs, R_sup=R_sup,
    #                         out_dir=out_dir)
