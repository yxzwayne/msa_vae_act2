#!/usr/bin/env python3

# Package import
import numpy as np
import pandas as pd
import h5py
import sys
import seqload

# Module import
from tqdm import tqdm
from numpy import random
from scipy.stats import pearsonr, spearmanr

npos = int(sys.argv[1])
reps = int(sys.argv[2])

targetseqs = seqload.loadSeqs(sys.argv[3], "ABCD")[0]
modelseqs = seqload.loadSeqs(sys.argv[4], "ABCD")[0]
indepseqs = seqload.loadSeqs(sys.argv[5], "ABCD")[0]
model_name = ""
if len(sys.argv) > 6:
    model_name = sys.argv[6]

targetseqs = (targetseqs + ord("A")).view("S1")
modelseqs = (modelseqs + ord("A")).view("S1")
indepseqs = (indepseqs + ord("A")).view("S1")

L = targetseqs.shape[1]
Nd = targetseqs.shape[0]
Np = modelseqs.shape[0]
Ni = indepseqs.shape[0]

allseq = np.concatenate([targetseqs, modelseqs, indepseqs])
choose_from = np.arange(0, L, 1)
choose_from = choose_from.astype("int")
store_path = "PI.h5"

repstack, su, fd2, fp2, fi2 = [], [], [], [], []
hist = lambda a, bins: np.histogram(a, bins=bins, density=True)[0].tolist()

for i in range(reps):
    chosenpos = random.choice(choose_from, npos, replace=False)

    # compute marginals (both datasets together, to we get counts for
    # subsequences that only appear in one datasets)
    # Numpy has no clear way to compute subsequence frequencies, so we
    # use a trick involving viewing the memory with a V(oid) datatype.
    v = np.ascontiguousarray(allseq[:, chosenpos]).view("S{}".format(npos))
    u, c = np.unique(v, return_inverse=True)
    bins = np.arange(len(u) + 1) - 0.5
    fd = hist(c[:Nd], bins)
    fp = hist(c[Nd : Nd + Np], bins)
    fi = hist(c[Nd + Np : Nd + Np + Ni], bins)

    fd2 += fd
    fp2 += fp
    fi2 += fi

    repstack += [i + 1 for n in range(0, len(u))]
    su += [
        int("".join([str("-ABCD".index(ss)) for ss in (str(s).split("'")[1])]))
        for s in u
    ]


result = np.vstack((repstack, su, fd2, fp2, fi2)).T
label = "order{}".format(npos)
model_colname = model_name if model_name else "model"
columns = map(
    lambda x: x,
    ["repnum", "subsequence", "data", model_colname, "indep"],
)
df = pd.DataFrame(result, columns=columns)

with pd.HDFStore(store_path) as store:
    store[label] = df


# ====== plotting

threshold = 0.02
orders = np.arange(2, 14)
min_order, max_order = 2, 13

pp = []
r = np.arange(1, 501)
for onum in orders:
    o = "/order" + str(onum)
    gg = df[o]["data"]
    gg2 = df[o]["indep"]
    gg3 = df[o][model_colname]
    gg4 = df[o]["repnum"]

    g = np.vstack((gg, gg2, gg3, gg4)).T
    gl = g[np.argsort(-g[:, 0])]
    h = gl[gl[:, 0] > threshold]

    if onum == min_order:
        gl_o = gl
        h_o = h
    pp.append([spearmanr(h[:, 0], h[:, 1])[0], spearmanr(h[:, 0], h[:, 2])[0]])

pp = np.asarray(pp)
