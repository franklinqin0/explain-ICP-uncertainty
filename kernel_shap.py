import shap
import scipy.special
import numpy as np
import itertools

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def shapley_kernel(M,s):
    if s == 0 or s == M:
        return 10000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

def f(X):
    # only consider kl div of rotation for now
    if np.array_equal(X, np.array([0., 1.])): # 0 0
        return 0
    elif np.array_equal(X, np.array([0.01, 1.])): # 1 0
        return 8.386546528335801
    elif np.array_equal(X, np.array([0., 1.2])): # 
        return 0.010302974173961754
    elif np.array_equal(X, np.array([0.01, 1.2])):
        return 9.195060856557436
    else:
        raise Exception("not expected")

def kernel_shap(f, x, reference, M):
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
        
    y = np.array([f(V[i, :]) for i in range(V.shape[0])])
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

M = 2
# np.random.seed(4)
# x = np.random.randn(M)
x = np.array([0.3, 1.5])
reference = np.array([1, 2]) # sensor noise from 0 to 2, init uncertainty from 1 to 3
# phi = kernel_shap(f, x, reference, M)
# base_value = phi[-1]
# shap_values = phi[:-1]

# print("  reference =", reference)
# print("          x =", x)
# print("shap_values =", shap_values)
# print(" base_value =", base_value)
# print("   sum(phi) =", np.sum(phi))
# print("       f(x) =", f(x))
# print("===")

# TODO: fix `f`
explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
shap_values = explainer.shap_values(x)
print("shap_values =", shap_values)
print("base value =", explainer.expected_value)
