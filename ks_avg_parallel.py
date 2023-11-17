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
import matplotlib.pyplot as plt
from multiprocessing import Pool

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


def worker_function(args):
    dataset, seq, scan_ref, scan_in, x, reference, M = args
    phi = kernel_shap(f, dataset, seq, scan_ref, scan_in, x, reference, M)
    shap_val = phi[:-1]
    return scan_ref, shap_val

if __name__ == "__main__":
    M = 3
    dataset = Dataset()
    seq = dataset.sequences[int(sys.argv[1])]
    print(seq)
    T_gt = dataset.get_data(seq)
    overlap_mat = dataset.get_overlap_matrix(seq)

    path_pickle = os.path.join(Param.path_sequence_base, seq, f'shap_{seq}.p')

    if not os.path.exists(path_pickle):
        mondict = {}
        tasks = []

        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            Param.curr_overlap = overlap_mat[scan_ref, scan_in]
            if Param.curr_overlap < 0.1:
                continue
            reference = np.array([0.0, 1.0, Param.curr_overlap])
            x = np.array([0.05, 1.5, Param.curr_overlap - 0.05])
            tasks.append((dataset, seq, scan_ref, scan_in, x, reference, M))

        with Pool() as pool:
            results = pool.map(worker_function, tasks)

        for scan_ref, shap_val in results:
            mondict[scan_ref] = shap_val

        dataset.dump(mondict, path_pickle)
