import os
import sys
from utils import *
import icp
from dataset import Dataset

def main(seq, sn, iu):
    Param.sensor_noise = sn
    Param.init_unc = iu
    Param.update()

    dataset = Dataset()
    T_gt = dataset.get_data(seq)
    print("T_gt:", T_gt.shape[0])
    for scan_ref in range(1):
        
        Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0))
        Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc))
        
        clean_path = os.path.join(Param.results_path, str(scan_ref))
        if not os.path.exists(clean_path):
            os.makedirs(clean_path)
        # add noise
        input_folder = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(0.0))
        output_folder = os.path.join(Param.path_sequence_base, seq, "local_frame", dec2str(Param.sensor_noise))
        if input_folder != output_folder and not os.path.exists(output_folder):
            print(f"Creating dataset with sensor noise {Param.sensor_noise} at {output_folder}")
            add_noise(Param.sensor_noise, input_folder, output_folder)
        
        pert_path = os.path.join(Param.results_pert, str(scan_ref))
        if not os.path.exists(pert_path):
            os.makedirs(pert_path)
        # get data for a pair of point clouds
        scan_in = scan_ref + 1
        icp.mc(dataset, pert_path, seq, scan_ref, scan_in)
        kl_div = icp.results(dataset, clean_path, pert_path, seq, scan_ref, scan_in)
        print("seq:", seq, "sn:", sn, "iu:", iu, "scan_ref:", scan_ref)
        print("rotation:", kl_div[0])
        print("translation:", kl_div[1])
        return kl_div[0], kl_div[1]

if __name__ == "__main__":
    seq = sys.argv[1]
    sn = float(sys.argv[2])
    iu = float(sys.argv[3])

    main(seq, sn, iu)
