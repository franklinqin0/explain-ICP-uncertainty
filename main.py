import os
from utils import *
import icp
from dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset()
    
    # create directories for clean and perturbed results
    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        for n in range(T_gt.shape[0]-1):
            path = os.path.join(Param.results_path, seq, str(n))
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(Param.results_pert, seq, str(n))
            if not os.path.exists(path):
                os.makedirs(path)
    
    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        # compute ICP covariance and results
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            # Monte-Carlo covariance
            icp.mc(dataset, Param.results_pert, seq, scan_ref, scan_in)
        
        # get ICP results
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            # compute results
            kl_div = icp.results(dataset, seq, scan_ref, scan_in)
            print("rotation:", kl_div[0])
            print("translation:", kl_div[1])
    
    # icp.aggregate_results(dataset)
