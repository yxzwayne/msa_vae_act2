#!/usr/bin/env python3
import numpy as np
from numpy import random
import pandas as pd
import h5py
import sys
import seqload
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import h5py, re, sys
import numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import UnivariateSpline
from pylab import *

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
N_target = targetseqs.shape[0]
N_model = modelseqs.shape[0]
N_indep = indepseqs.shape[0]

allseq = np.concatenate([targetseqs, modelseqs, indepseqs])
choose_from = np.arange(0, L, 1)
choose_from = choose_from.astype("int")
store_path = f"{model_name}_PI.h5"

repstack, su, fd2, fp2, fi2 = [], [], [], [], []
hist = lambda a, bins: np.histogram(a, bins=bins, density=True)[0].tolist()

for i in tqdm(range(reps)):
    chosenpos = random.choice(choose_from, npos, replace=False)

    # compute marginals (both datasets together, to we get counts for
    # subsequences that only appear in one datasets)
    # Numpy has no clear way to compute subsequence frequencies, so we
    # use a trick involving viewing the memory with a V(oid) datatype.
    subseqs = np.ascontiguousarray(allseq[:, chosenpos])
    v = np.ascontiguousarray(allseq[:, chosenpos]).view("S{}".format(npos))
    u, c = np.unique(v, return_inverse=True)
    bins = np.arange(len(u) + 1) - 0.5
    fd = hist(c[:N_target], bins)
    fp = hist(c[N_target : N_target + N_model], bins)
    fi = hist(c[N_target + N_model : N_target + N_model + N_indep], bins)

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
columns = map(lambda x: x, ["repnum", "subsequence", "data", "model", "indep"])
df = pd.DataFrame(result, columns=columns)

with pd.HDFStore(store_path) as store:
    store[label] = df

# ======= originally plot_comparison3.py

threshold = 0.02
orders = np.arange(2, 14)
min_order, max_order = 2, 13

pp = []
r = np.arange(1, 501)

with pd.HDFStore("PI.h5") as store:
    for onum in orders:
        # f=open('pp'+str(onum),"w")
        o = "/order" + str(onum)
        # if d.data>0:
        gg = store[o]["data"]
        gg2 = store[o]["indep"]
        gg3 = store[o]["model"]
        gg4 = store[o]["repnum"]

        g = np.vstack((gg, gg2, gg3, gg4)).T
        # g=g[g[:,0]>threshold]
        gl = g[np.argsort(-g[:, 0])]
        h = gl[gl[:, 0] > threshold]

        # h=[gl[gl[:,3]==rr][1] for rr in r]
        # h=np.asarray(h)
        if onum == min_order:
            gl_o = gl
            h_o = h
        pp.append([spearmanr(h[:, 0], h[:, 1])[0], spearmanr(h[:, 0], h[:, 2])[0]])
        # pp.append([spearmanr(gl[:,0],gl[:,1])[0],spearmanr(gl[:,0],gl[:,2])[0]])

pp = np.asarray(pp)

# np.savetxt('pi.spearman.each_secondtop.csv',pp)


def interpolate_curve(xs, ys, steps=200):
    xs_smooth = np.linspace(xs.min(), xs.max(), steps)
    spline = UnivariateSpline(xs, ys, k=3, s=0.03)
    new_ys = spline(xs_smooth)
    return xs_smooth, new_ys


oi, pi = interpolate_curve(orders, pp[:, 1] ** 2)
oi, ii = interpolate_curve(orders, pp[:, 0] ** 2)

plot(orders, pp[:, 1] ** 2, "o", color="royalblue")
plot(orders, pp[:, 0] ** 2, "o", color="gray")
plot(oi, pi, color="royalblue", lw=3, label="Potts")
plot(oi, ii, color="gray", lw=3, label="Indep")
axhline(1, ls="dashed", lw=1.5, alpha=0.6)
legend(loc="lower left", fontsize=12)
axes = gca()
axes.set_ylim([0, 1.05])
# axes.set_xlim([1.5,14.5])
tick_params(axis="both", length=5, width=2, labelsize=10)
ylabel(r"Spearman $\rho^2$ for marginals > " + str(threshold), fontsize=12)
xlabel(r"Subsequence length", fontsize=15)
tight_layout()
show()

plot(gl[:, 0], gl[:, 1], ".", color="gray", alpha=0.6, label="Indep")
plot(gl[:, 0], gl[:, 2], ".", color="royalblue", alpha=0.6, label="Potts")
plot(h[:, 0], h[:, 1], ".", color="k", alpha=0.4, label="Indep above cutoff")
plot(h[:, 0], h[:, 2], ".", color="darkblue", alpha=0.8, label="Potts above cutoff")
legend(loc="upper left", fontsize=12)
tick_params(axis="both", length=5, width=2, labelsize=10)
ylabel(r"Model prediction", fontsize=15)
xlabel(r"Data marginal >" + str(threshold), fontsize=15)
tight_layout()
show()

plot(gl_o[:, 0], gl_o[:, 1], ".", color="gray", alpha=0.6, label="Indep")
plot(gl_o[:, 0], gl_o[:, 2], ".", color="royalblue", alpha=0.6, label="Potts")
plot(h_o[:, 0], h_o[:, 1], ".", color="k", alpha=0.4, label="Indep above cutoff")
plot(h_o[:, 0], h_o[:, 2], ".", color="darkblue", alpha=0.8, label="Potts above cutoff")
legend(loc="upper left", fontsize=12)
tick_params(axis="both", length=5, width=2, labelsize=10)
ylabel(r"Model prediction", fontsize=15)
xlabel(r"Data marginal >" + str(threshold), fontsize=15)
tight_layout()
show()

print(pp**2)
