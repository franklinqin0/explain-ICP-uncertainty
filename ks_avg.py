import os
import sys
import random
import scipy.special
import numpy as np
import itertools
from utils import Param
from dataset import Dataset
from uncertainty import uncertainty
import shap
from multiprocessing import Pool
import matplotlib.pyplot as plt

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

def f(dataset, seq, scan_ref, scan_in, X):
    return uncertainty(dataset, seq, scan_ref, scan_in, *X)

def kernel_shap(f, dataset, seq, scan_ref, scan_in, x, reference, M):
    X = np.zeros((2**M,M+1))
    X[:,-1] = 1
    weights = np.zeros(2**M)
    V = np.zeros((2**M,M))
    for i in range(2**M):
        V[i, :] = reference

    for i,s in enumerate(powerset(range(M))):
        s = list(s)
        V[i,s] = x[s]
        X[i,s] = 1
        weights[i] = shapley_kernel(M, len(s))
        
    y = np.array([f(dataset, seq, scan_ref, scan_in, V[i, :]) for i in range(V.shape[0])])
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

M = 3
dataset = Dataset()


seq = dataset.sequences[int(sys.argv[1])]
print(seq)
T_gt = dataset.get_data(seq)
overlap_mat = dataset.get_overlap_matrix(seq)

# compute ICP covariance and results
path_pickle = os.path.join(Param.path_sequence_base, seq, f'shap_{seq}.p')

if os.path.exists(path_pickle):
        mondict = dataset.load(path_pickle)
        print(mondict) # TODO
else:
    mondict = {}
    for scan_ref in range(T_gt.shape[0]-1):
        scan_in = scan_ref + 1
    
        Param.curr_overlap = overlap_mat[scan_ref, scan_in]
        if Param.curr_overlap < 0.1:
            raise Exception("curr overlap too small, change scan_ref / scan_in!")
        reference = np.array([0.0, 1.0, Param.curr_overlap])
        uncertainty(dataset, seq, scan_ref, scan_in, *reference)

        sn = 0.05
        iu = 1.5
        po = 0.05
        
        sensor_noise = round(sn, 2)
        init_unc = round(iu, 1)
        target_overlap = round(Param.curr_overlap - po, 2)
        x = np.array([sensor_noise, init_unc, target_overlap])
    
        phi = kernel_shap(f, dataset, seq, scan_ref, scan_in, x, reference, M)
        # base_val = phi[-1]
        shap_val = phi[:-1]
        print("shap val:", shap_val)

        mondict[scan_ref] = shap_val
    dataset.dump(mondict, path_pickle)
