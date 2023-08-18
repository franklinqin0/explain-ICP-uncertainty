import os
from utils import *
import icp
from dataset import Dataset

def uncertainty(scan_ref=0):
    # TODO: params: sensor_noise, init_uncertainty
    dataset = Dataset()

    # TODO: for now, run only on Apartment
    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        # for n in range(T_gt.shape[0]-1):
        n = 0
        path = os.path.join(Param.results_path, seq, str(n))
        if not os.path.exists(path):
            os.makedirs(path)
        # Param.results_pert = 
        path = os.path.join(Param.results_pert, seq, str(n))
        if not os.path.exists(path):
            os.makedirs(path)
        # get data for a pair of point clouds
        # scan_ref = 0
        scan_in = scan_ref + 1
        icp.mc(dataset, Param.results_pert, seq, scan_ref, scan_in)
        kl_div = icp.results(dataset, seq, scan_ref, scan_in)
        print("rotation:", kl_div[0])
        # print("translation:", kl_div[1])
        return kl_div[0]

uncertainty()