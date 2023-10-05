import os
import numpy as np
from utils import *

def mc(dataset, base_path, sequence, scan_ref, scan_in, curr_overlap):
    # base_path = os.path.join(path, sequence, str(scan_ref))
    if curr_overlap is None:
        filename = "mc_" + str(Param.n_mc-1) + ".txt"
    else:
        filename = "momc_" + str(Param.n_mc-1) + ".txt"
    path = os.path.join(base_path, filename)
    
    if os.path.exists(path):
        return
    T_gt = dataset.get_data(sequence)
    pc_ref, pc_in = dataset.get_pc(sequence, scan_ref, scan_in, curr_overlap)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])

    # Monte-Carlo
    for n in range(Param.n_mc):
        if curr_overlap is None:
            filename = "mc_" + str(n) + ".txt"
        else:
            filename = "momc_" + str(n) + ".txt"
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            # print(path + " already exist")
            continue

        # sample initial transformation
        T_init_n = SE3.normalize(SE3.mul(SE3.exp(-Param.xi[n]), T_init))  # T = exp(xi) T_hat
        icp_without_cov(pc_ref, pc_in, T_init_n, path)


def results(dataset, clean_path, pert_path, sequence, scan_ref, scan_in, curr_overlap):
    # base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    # f_metrics = os.path.join(base_path, 'metrics.p')
    # print("f_metrics: ", f_metrics)
    # if not Param.b_cov_results and os.path.exists(f_metrics):
    #     print(f_metrics + " already exists")
    #     return
    # cov_path = os.path.join(base_path, "cov_censi.txt")
    T_gt = dataset.get_data(sequence)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    T_mc, _ = dataset.get_mc_results(clean_path, None)
    Tp_mc, _ = dataset.get_mc_results(pert_path, curr_overlap)
    # T_ut, T_init_ut = dataset.get_ut_results(sequence, scan_ref)

    # _, _, cov_ut, cov_cross = ut_class.unscented_transform_se3(T_ut)
    
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
        mcp_new[n] = SE3.log(SE3.mul(Tp_mc[n], T_init_inv))  # xi = log( T * T_hat^{-1} )
    cov_mcp = np.cov(mcp_new.T)
    
    kl_div_mc = np.zeros((Param.n_mc, 2))

    for n in range(Param.n_mc):
        # kl_div_sensor[n] = rot_trans_kl_div(cov_mc, cov_sensor)
        kl_div_mc[n] = rot_trans_kl_div(cov_mc, cov_mcp)

    # kl_div_mc_avg = np.sum(kl_div_mc, 0) / Param.n_mc
    
    # kl_div_tgt_avg = np.sum(kl_div(cov_mc, cov_mcp)) / Param.n_mc
    # print("kl_div_tgt_avg:", kl_div_tgt_avg)
    
    # print("kl_div_mc_avg:", kl_div_mc_avg)
    # return kl_div_mc_avg
    # print("kl_div_tgt_avg:", kl_div_tgt_avg)
    
    kl_div_mc_med = np.median(kl_div_mc, axis=0, overwrite_input=True)
    # kl_div_tgt_med = np.median(kl_div(cov_mc, cov_mcp), axis=None, overwrite_input=True)
    print("kl_div_mc_med:", kl_div_mc_med)
    return kl_div_mc_med
    # print("kl_div_tgt_med:", kl_div_tgt_med)
    # print("===")
