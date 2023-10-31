import os
import numpy as np
from utils import *

def mc(dataset, base_path, sequence, scan_ref, scan_in, target_overlap):
    # base_path = os.path.join(path, sequence, str(scan_ref))
    filename = "mc_" + str(Param.n_mc-1) + ".txt"
    path = os.path.join(base_path, filename)
    
    if os.path.exists(path):
        return
    T_gt = dataset.get_data(sequence)
    pc_ref, pc_in = dataset.get_pc(sequence, scan_ref, scan_in)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])

    # Monte-Carlo
    for n in range(Param.n_mc):
        filename = "mc_" + str(n) + ".txt"
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            # print(path + " already exist")
            continue

        # sample initial transformation
        T_init_n = SE3.normalize(SE3.mul(SE3.exp(-Param.xi[n]), T_init))  # T = exp(xi) T_hat
        icp_without_cov(pc_ref, pc_in, T_init_n, path)


def results(dataset, clean_path, pert_path, sequence, scan_ref, scan_in):
    # base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    # f_metrics = os.path.join(base_path, 'metrics.p')
    # print("f_metrics: ", f_metrics)
    # if not Param.b_cov_results and os.path.exists(f_metrics):
    #     print(f_metrics + " already exists")
    #     return
    T_gt = dataset.get_data(sequence)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    T_mc, _ = dataset.get_mc_results(clean_path)
    Tp_mc, _ = dataset.get_mc_results(pert_path)
    
    # unperturbed Monte-Carlo errors
    mc_new = np.zeros((Param.n_mc, 6))
    T_init_inv = SE3.inv(T_init)
    for n in range(Param.n_mc):
        mc_new[n] = SE3.log(SE3.mul(T_mc[n], T_init_inv))  # xi = log( T * T_hat^{-1} )
    cov_mc = np.cov(mc_new.T)
    
    # perturbed Monte-Carlo errors
    mcp_new = np.zeros((Param.n_mc, 6))
    T_init_inv = SE3.inv(T_init)
    for n in range(Param.n_mc):
        mcp_new[n] = SE3.log(SE3.mul(Tp_mc[n], T_init_inv))
    cov_mcp = np.cov(mcp_new.T)
    
    kl_div_mc = kl_div(cov_mc, cov_mcp)
    return kl_div_mc
