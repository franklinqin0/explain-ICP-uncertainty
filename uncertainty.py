import os
import sys
from utils import *
import icp
from dataset import Dataset
from overlap import add_noise

def uncertainty(dataset, seq, scan_ref, scan_in, sn, iu, po):
    Param.sensor_noise = round(sn, 2)
    Param.init_unc = round(iu, 1)
    Param.update()
    target_overlap = round(po, 3)

    T_gt = dataset.get_data(seq)

    if (abs(target_overlap - Param.curr_overlap) < 1e-3):
        Param.path_pc = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(Param.sensor_noise))
        Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0))
        Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc))
    else: # need to adjust overlap
        Param.path_pc = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(Param.sensor_noise), dec2str(target_overlap))
        Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0), dec2str(Param.curr_overlap))
        Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc), dec2str(target_overlap))
        
    clean_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0), str(scan_ref))
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    pert_path = os.path.join(Param.results_pert, str(scan_ref))
    if not os.path.exists(pert_path):
        os.makedirs(pert_path)
    
    clean_input_folder = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(0.0))

    print(f"Calculating uncertainty for: sensor noise = {Param.sensor_noise} and init uncertainty = {Param.init_unc} with partial overlap = {target_overlap!s:^5}")
    # add noise and adjust overlap if needed
    add_noise(clean_input_folder, Param.path_pc, scan_ref, scan_in, Param.sensor_noise, T_gt, target_overlap)

    # assume that `clean_path` has Param.n_mc runs already
    # run ICP algorithm `Param.n_mc` times for `pert_path`
    icp.mc(dataset, pert_path, seq, scan_ref, scan_in, target_overlap)
    # calculate results
    kl_div = icp.results(dataset, clean_path, pert_path, seq, scan_ref, scan_in)
    print("KL div:", kl_div)
    return kl_div

if __name__ == "__main__":
    dataset = sys.argv[1]
    seq = sys.argv[2]
    scan_ref = sys.argv[3]
    scan_in = sys.argv[4]
    sn = float(sys.argv[5])
    iu = float(sys.argv[6])
    po = float(sys.argv[7])
    
    # dataset = Dataset()
    # seq = "Apartment"
    # scan_ref = 1
    # scan_in = 2
    # sn = 0.0
    # iu = 1.0
    # Param.curr_overlap = 0.872
    # po = Param.curr_overlap - 0.1
    # # po = 0.850
    
    uncertainty(dataset, seq, scan_ref, scan_in, sn, iu, po)
