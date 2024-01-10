#!/usr/bin/env python
import pandas as pd
import h5py, re, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import UnivariateSpline
from pylab import *

# threshold=float(sys.argv[1])
threshold = 0.02
marg_store = "PI.h5"
# marg_store='../pairwise_nonPI.h5'
# marg_store='../multimarg_0.02.h5'
orders = np.arange(2, 14)
# orders=np.delete(orders,11)
min_order, max_order = 2, 13

pp = []
r = np.arange(1, 501)
with pd.HDFStore(marg_store) as store:
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
