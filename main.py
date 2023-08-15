import os

import icp
from dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset()
    
    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        for n in range(T_gt.shape[0]-1):
            path = os.path.join("results", seq, str(n))
            if not os.path.exists(path):
                os.makedirs(path)
    
    # for seq in dataset.sequences:
        # T_gt = dataset.get_data(sequence)
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            # should have computed Monte-Carlo results
            # get ICP covariance
            icp.results(dataset, seq, scan_ref, scan_in)
            