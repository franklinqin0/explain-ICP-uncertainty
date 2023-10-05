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
    po = round(po, 3)

    T_gt = dataset.get_data(seq)

    Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0))
    Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc))
    
    clean_path = os.path.join(Param.results_path, str(scan_ref))
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    
    # add noise
    clean_input_folder = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(0.0))
    pert_input_folder = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(Param.sensor_noise))
    
    curr_overlap = None
    if (abs(po - Param.mean_overlap) < 1e-3):
        # current overlap ratio is set to mean overlap
        curr_overlap = po

    print(f"Creating dataset with sensor noise = {Param.sensor_noise} and init uncertainty = {Param.init_unc} with current overlap = {curr_overlap!s:^4}")
    add_noise(clean_input_folder, pert_input_folder, scan_ref, scan_in, Param.sensor_noise, T_gt, curr_overlap)

    pert_path = os.path.join(Param.results_pert, str(scan_ref))
    if not os.path.exists(pert_path):
        os.makedirs(pert_path)

    # get data for a pair of point clouds
    # scan_in = scan_ref + 1
    # assume that `clean_path` has Param.n_mc runs already
    # run ICP algorithm `Param.n_mc` times
    icp.mc(dataset, pert_path, seq, scan_ref, scan_in, curr_overlap)
    # calculate results
    kl_div = icp.results(dataset, clean_path, pert_path, seq, scan_ref, scan_in, curr_overlap)
    # print("sn:", sn, "iu:", iu, "scan_ref:", scan_ref)
    # print("rotation:", kl_div)
    # print("translation:", kl_div[1])
    # TODO: decide if want translation, or combine the two values
    return kl_div[0] # , kl_div[1]

if __name__ == "__main__":
    # dataset = sys.argv[1]
    # seq = sys.argv[2]
    # scan_ref = sys.argv[3]
    # scan_in = sys.argv[4]
    # sn = float(sys.argv[5])
    # iu = float(sys.argv[6])
    # po = float(sys.argv[7])
    
    dataset = Dataset()
    seq = "Apartment"
    scan_ref = 1
    scan_in = 2
    sn = 0.03
    iu = 1.1
    # po = 0.95
    po = 0.850
    
    uncertainty(dataset, seq, scan_ref, scan_in, sn, iu, po)
