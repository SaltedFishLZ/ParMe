"""Generate random daya for the cocktail model
"""
import json
import pickle

import numpy as np


def data_gen(r, l, m, seed=0):

    # create problem
    name = "random.{:03d}".format(seed)
    np.random.seed(seed)

    # resource usage
    R = np.random.randint(low=0, high=64, size=(r, l), dtype=int)
    print("=" * 64)
    print("Resource usage (original)")
    print(R)

    # volume for each type of design (l designs in total)
    v =  np.random.randint(low=100, high=401, size=l, dtype=int)
    print("=" * 64)
    print("Product volume (original)")
    print(v)

    # unit cost for each type of resources
    c = np.random.uniform(low=0.5, high=13.3, size=r)
    print("=" * 64)
    print("Resource unit cost (original)")
    print(c)

    meta = {
        # meta-data
        "r" : r, "l" : l, "m" : m,
    }

    data = {
        # real data
        "R" : R, "v" : v, "c" : c
    }

    # dump all meta data
    fname = name + ".meta.dat"
    with open(fname, "w") as fout:
        json.dump(meta, fout)

    # dump all data
    for key in data.keys():
        fname = name + "." + key + ".dat"
        np.savetxt(fname, data[key])


    # dump all data into binary pickle file
    fname = name + ".pkl"
    with open(fname, "wb") as fout:
        pickle.dump({**meta, **data}, fout)




if __name__ == "__main__":

    # r = 4, m = 5
    r, l, m = 4, 10, 5
    data_gen(r=r, l=l, m=m, seed=0)

    r, l, m = 4, 20, 5
    data_gen(r=r, l=l, m=m, seed=1)

    r, l, m = 4, 40, 5
    data_gen(r=r, l=l, m=m, seed=2)

    # r = 4, m = 2
    r, l, m = 4, 10, 2
    data_gen(r=r, l=l, m=m, seed=3)

    r, l, m = 4, 20, 2
    data_gen(r=r, l=l, m=m, seed=4)

    r, l, m = 4, 40, 2
    data_gen(r=r, l=l, m=m, seed=5)

    # r = 8, m = 5
    r, l, m = 8, 10, 5
    data_gen(r=r, l=l, m=m, seed=6)

    r, l, m = 8, 20, 5
    data_gen(r=r, l=l, m=m, seed=7)

    r, l, m = 8, 40, 5
    data_gen(r=r, l=l, m=m, seed=8)