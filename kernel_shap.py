import scipy.special
import numpy as np
import itertools
from collections import OrderedDict
from utils import Param
from dataset import Dataset
from uncertainty import uncertainty
from overlap import mean_overlap

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

scan_ref = 1
scan_in = scan_ref + 1
seq = "Apartment"

M = 3
dataset = Dataset()
shap = OrderedDict()

overlap_mat = dataset.get_overlap_matrix(seq)
curr_overlap_ratio = overlap_mat[scan_ref, scan_in]
Param.mean_overlap = mean_overlap(scan_ref, scan_in, overlap_mat)
reference = np.array([Param.mean_noise, Param.mean_unc, Param.mean_overlap])
# reference = np.array([0.0, 1.0, Param.mean_overlap])

uncertainty(dataset, seq, scan_ref, scan_in, 0.0, 1.0, curr_overlap_ratio)

for sn in np.arange(0.0, 0.101, 0.01):
    for iu in np.arange(1.0, 2.001, 0.1):
        curr_sensor_noise = round(sn, 2)
        curr_init_unc = round(iu, 1)
        x = np.array([curr_sensor_noise, curr_init_unc, curr_overlap_ratio])
        
        phi = kernel_shap(f, dataset, seq, scan_ref, scan_in, x, reference, M)
        base_value = phi[-1]
        shap_values = phi[:-1]
        print("shap_values:", shap_values)
        shap[(curr_sensor_noise, curr_init_unc)] = shap_values

print(shap)
