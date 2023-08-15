import os
import numpy as np
from utils import *

# def mc(dataset, sequence, scan_ref, scan_in):
    

def results(dataset, sequence, scan_ref, scan_in):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    f_metrics = os.path.join(base_path, 'metrics.p')
    # print("f_metrics: ", f_metrics)
    # if not Param.b_cov_results and os.path.exists(f_metrics):
    #     print(f_metrics + " already exists")
    #     return
    # cov_path = os.path.join(base_path, "cov_censi.txt")
    T_gt = dataset.get_data(sequence)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    T_mc, T_init_mc = dataset.get_mc_results(sequence, scan_ref)
    # T_ut, T_init_ut = dataset.get_ut_results(sequence, scan_ref)

    # _, _, cov_ut, cov_cross = ut_class.unscented_transform_se3(T_ut)
    
    # Monte-Carlo errors
    mc_new = np.zeros((Param.n_mc, 6))
    T_init_inv = SE3.inv(T_init)
    for n in range(Param.n_mc):
        mc_new[n] = SE3.log(SE3.mul(T_mc[n], T_init_inv))  # xi = log( T * T_hat^{-1} )
    cov_mc = np.cov(mc_new.T)
    
    # cov_sensor = 
    
    kl_div_mc = np.zeros((Param.n_mc, 2))
    
    for n in range(1): # Param.n_mc):
        # kl_div_sensor[n] = rot_trans_kl_div(cov_mc, cov_sensor)
        kl_div_mc[n] = rot_trans_kl_div(cov_mc, cov_mc)
    
    kl_div_mc = np.sum(kl_div_mc, 0) / Param.n_mc
    
    kl_div_tgt = np.sum(kl_div(cov_mc, cov_mc)) / Param.n_mc
    print("kl_div_mc:", kl_div_mc)
    print("kl_div_tgt:", kl_div_tgt)