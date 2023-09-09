import os
from utils import *
import icp
from dataset import Dataset


Param.sensor_noise = 2.1
Param.init_unc = 3.1
Param.update()

def uncertainty(): # scan_ref=0
    dataset = Dataset()

    # TODO: for now, run only on Apartment
    for seq in dataset.sequences:
        T_gt = dataset.get_data(seq)
        for scan_ref in range(T_gt.shape[0]-1):
            for sn in np.arange(0, 2.1, 0.1):
                for iu in np.arange(1, 3.1, 0.1):
                    Param.sensor_noise = sn
                    Param.init_unc = iu
                    Param.update()
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
                    print("rotation:", kl_div[0])
                    print("translation:", kl_div[1])
                    # return kl_div[0], kl_div[1]

rot, trans = uncertainty()
# print("rot:", rot)
# print("trans:", trans)
