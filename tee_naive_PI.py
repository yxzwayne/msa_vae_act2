#!/usr/bin/env python3
import numpy as np
from numpy import random
import pandas as pd
import h5py
import sys
import seqload
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

npos = int(sys.argv[1])
print("Processing marginal order ", npos)
reps = int(sys.argv[2])

dataseqs = seqload.loadSeqs(sys.argv[3],'ABCD')[0]
modelseqs = seqload.loadSeqs(sys.argv[4],'ABCD')[0]
indepseqs = seqload.loadSeqs(sys.argv[5],'ABCD')[0]

dataseqs = (dataseqs + ord('A')).view('S1')
modelseqs = (modelseqs + ord('A')).view('S1')
indepseqs = (indepseqs + ord('A')).view('S1')
#w = np.load(sys.argv[6])

L = dataseqs.shape[1]
Nd = dataseqs.shape[0]
Np = modelseqs.shape[0]
Ni = indepseqs.shape[0]

allseq = np.concatenate([dataseqs, modelseqs, indepseqs])
# choose_from = np.loadtxt('PI_assoc_pos',dtype='int')
choose_from = np.arange(0, L, 1).astype(int)
store_path = 'PI.h5'

# f1 = open('multimarg/drug_naive/PI/Potts_indep_'+str(npos),"w")
# f2 = open('multimarg/drug_naive/PI/chosen_pos_'+str(npos),"w")

repstack,posstack,su,fd2,fp2,fi2=[],[],[],[],[],[]
hist = lambda a, bins: np.histogram(a, bins=bins, density=True)[0].tolist()

for i in tqdm(range(reps)):
    #chosenpos = choose_from[i]
    chosenpos = random.choice(choose_from, npos, replace=False)
    pos_string = int(''.join([str(i+1) for i in chosenpos]))

    # compute marginals (both datasets together, to we get counts for
    # subsequences that only appear in one datasets)
    # Numpy has no clear way to compute subsequence frequencies, so we
    # use a trick involving viewing the memory with a V(oid) datatype.
    subseqs = np.ascontiguousarray(allseq[:,chosenpos])
    v = np.ascontiguousarray(allseq[:,chosenpos]).view('S{}'.format(npos))
    u, c = np.unique(v, return_inverse=True)
    bins = np.arange(len(u)+1)-0.5
    fd = hist(c[:Nd],bins)
    fp = hist(c[Nd:Nd+Np],bins)
    fi = hist(c[Nd+Np:Nd+Np+Ni],bins)

    fd2 +=fd
    fp2 +=fp
    fi2 +=fi

    repstack +=[i+1 for n in range(0,len(u))]
    #posstack +=[pos_string for n in range(0,len(u))]
    su +=[int(''.join([str('-ABCD'.index(ss)) for ss in (str(s).split("'")[1])])) for s in u]

    # print(i+1,[chosenpos+1][0],file=f2)

    # top20 = np.argsort(fd)[-20:]
    # fd20 = [fd[m] for m in top20]
    # fp20 = [fp[m] for m in top20]
    # fi20 = [fi[m] for m in top20]
    
    # if len(top20) < 2:
    #     print(0,file=f1)
    # else:
    #     print(pearsonr(fd20, fp20)[0],(pearsonr(fd20, fi20))[0], file=f1)


result=np.vstack((repstack, su, fd2, fp2, fi2)).T
label='order{}'.format(npos)
columns=map(lambda x: x, ['repnum','subsequence','data','model','indep'])
df=pd.DataFrame(result, columns=columns)

with pd.HDFStore(store_path) as store:
    store[label]=df

# f1.close()
# f2.close()
