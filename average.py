import time
import numpy as np
from dataset import Dataset
from utils import Param
from multiprocessing import Pool
from uncertainty import uncertainty

def uncertainty_wrapper(args):
    return uncertainty(*args)


if __name__ == "__main__":
    M = 3
    dataset = Dataset()

    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        overlap_mat = dataset.get_overlap_matrix(seq)
        # compute ICP covariance and results
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            Param.curr_overlap = overlap_mat[scan_ref, scan_in]
            if Param.curr_overlap < 0.1:
                raise Exception("curr overlap too small, change scan_ref / scan_in!")
            reference = np.array([0.0, 1.0, Param.curr_overlap])
            uncertainty(dataset, seq, scan_ref, scan_in, *reference)

            params_list = []
            
            # for sn in np.arange(0.0, 0.101, 0.01):
            #     for iu in np.arange(1.0, 2.001, 0.1):
            #         for po in np.arange(0.0, 0.101, 0.01):
            
            sn = 0.05
            iu = 1.5
            po = 0.05
            
            sensor_noise = round(sn, 2)
            init_unc = round(iu, 1)
            target_overlap = round(Param.curr_overlap - po, 2)
            x = np.array([sensor_noise, init_unc, target_overlap])
            params_list.append((dataset, seq, scan_ref, scan_in, sensor_noise, init_unc, target_overlap))

    print("Starting!")
    start = time.time()
    with Pool() as pool:
        pool.map(uncertainty_wrapper, params_list)
    end = time.time()
    print("Finished! Total running time is:", end-start)
