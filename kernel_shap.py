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

# The following code comes from the simple kernel SHAP implementation:
# https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/Simple%20Kernel%20SHAP.html

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

scan_ref = int(sys.argv[1])
scan_in = scan_ref + 1
seq = "Apartment"
M = 3
dataset = Dataset(read_data=False)

shap_values = []
features = []
path_pickle = os.path.join(Param.path_sequence_base, seq, f'shap_{scan_ref}_{scan_in}.p')
overlap_mat = dataset.get_overlap_matrix(seq)
Param.curr_overlap = overlap_mat[scan_ref, scan_in]

if os.path.exists(path_pickle):
    mondict = dataset.load(path_pickle)
    shap_values, features = mondict['shap_values'], mondict['features']
else:
    if Param.curr_overlap < 0.1:
        raise Exception("curr overlap too small, change scan_ref / scan_in!")
    reference = np.array([0.0, 1.0, Param.curr_overlap])
    uncertainty(dataset, seq, scan_ref, scan_in, *reference)
    
    for sn in np.arange(0.0, 0.101, 0.01):
        for iu in np.arange(1.0, 2.001, 0.1):
            for po in np.arange(0.0, 0.101, 0.01):
                
                sensor_noise = round(sn, 2)
                init_unc = round(iu, 1)
                target_overlap = round(Param.curr_overlap - po, 2)
                x = np.array([sensor_noise, init_unc, target_overlap])
                
                phi = kernel_shap(f, dataset, seq, scan_ref, scan_in, x, reference, M)
                # base_val = phi[-1]
                shap_val = phi[:-1]
                print("shap val:", shap_val)
                shap_values.append(shap_val)
                features.append([sensor_noise, init_unc, target_overlap])

    mondict = {'shap_values': shap_values, 'features': features}
    dataset.dump(mondict, path_pickle)

# start visualization
feature_names = ['sensor_noise', 'init_pose', 'partial_overlap']
shap_values = np.asarray(shap_values)

features = np.asarray(features)
features[:, 2] = np.round(Param.curr_overlap - features[:, 2], 2)

# summary plot
shap.summary_plot(shap_values, features, feature_names, sort=False)
plt.savefig(f"summary_{scan_ref}_{scan_in}.png", bbox_inches='tight', dpi=600)

# dependence plot
shap.dependence_plot("sensor_noise", shap_values, features, feature_names=feature_names, interaction_index="init_pose")
plt.savefig(f"dependence_{scan_ref}_{scan_in}_sn_vs_ip.png", bbox_inches='tight', dpi=600)
shap.dependence_plot("sensor_noise", shap_values, features, feature_names=feature_names, interaction_index="partial_overlap")
plt.savefig(f"dependence_{scan_ref}_{scan_in}_sn_vs_po.png", bbox_inches='tight', dpi=600)
shap.dependence_plot("init_pose", shap_values, features, feature_names=feature_names, interaction_index="partial_overlap")
plt.savefig(f"dependence_{scan_ref}_{scan_in}_ip_vs_po.png", bbox_inches='tight', dpi=600)
shap.dependence_plot("partial_overlap", shap_values, features, feature_names=feature_names, interaction_index="init_pose")
plt.savefig(f"dependence_{scan_ref}_{scan_in}_po_vs_ip.png", bbox_inches='tight', dpi=600)

# waterfall plot
expl = shap.Explanation(values=shap_values, data=features, feature_names=feature_names, base_values=0.0)
idx = 1109
plt.clf()
shap.waterfall_plot(expl[idx], show=False)
plt.savefig(f"waterfall_{scan_ref}_{scan_in}_idx_{idx}.png", bbox_inches='tight', dpi=600)
idx = 1110
plt.clf()
shap.waterfall_plot(expl[idx], show=False)
plt.savefig(f"waterfall_{scan_ref}_{scan_in}_idx_{idx}.png", bbox_inches='tight', dpi=600)
