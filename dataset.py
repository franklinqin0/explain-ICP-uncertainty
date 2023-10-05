import os
import numpy as np
import pickle
import glob
from utils import *

class Dataset:
    sequences = [
        'Apartment',
        # 'Hauptgebaude',
        # 'Stairs',
        # 'Mountain',
        # 'Gazebo_summer',
        # 'Gazebo_winter',
        # 'Wood_summer',
        # 'Wood_winter',
    ]
    overlap_matrix = None

    def __init__(self):
        self.read_data()

    def read_data(self):
        for sequence in self.sequences:
            self.preprocessing_data(sequence)

    def preprocessing_data(self, sequence):
        path_sequence = os.path.join(Param.path_sequence_base, sequence)
        path_pickle = os.path.join(path_sequence, 'data.p')
        if os.path.exists(path_pickle):
            print('Data already preprocessed')
        else:
            # files = os.listdir(os.path.join(path_sequence, 'local_frame', dec2str(0.0)))
            path_pattern = os.path.join(path_sequence, 'local_frame', 'Hokuyo*', dec2str(0.0))
            files = glob.glob(path_pattern)
            n_scan = int((len(files)-1)/4)
            T_gt_file = os.path.join(path_sequence, 'global_frame', 'pose_scanner_leica.csv')
            T_gt_data = np.genfromtxt(T_gt_file, delimiter=',', skip_header=1)
            T_gt = SE3.new(n_scan)

            for k in range(3):
                T_gt[:, k, :] = T_gt_data[:, 2+4*k:6+4*k]

            overlap_matrix_path = os.path.join(Param.path_sequence_base, sequence, f"overlap_{sequence}.csv")
            overlap_matrix = np.genfromtxt(overlap_matrix_path, delimiter=',')
            mondict = {
                'T_gt': T_gt,
                f"overlap_{sequence}.csv": overlap_matrix
                }
            self.dump(mondict, path_pickle)

    def get_data(self, sequence):
        path_sequence = os.path.join(Param.path_sequence_base, sequence)
        path = os.path.join(path_sequence, 'data.p')
        mondict = self.load(path)
        return mondict['T_gt']

    def get_overlap_matrix(self, sequence):
        path_sequence = os.path.join(Param.path_sequence_base, sequence)
        path = os.path.join(path_sequence, 'data.p')
        mondict = self.load(path)
        return mondict[f"overlap_{sequence}.csv"]

    @classmethod
    def load(cls, *_file_name):
        file_name = os.path.join(*_file_name)
        with open(file_name, "rb") as file_pi:
            pickle_dict = pickle.load(file_pi)
        return pickle_dict

    @classmethod
    def dump(cls, mondict, *_file_name):
        file_name = os.path.join(*_file_name)
        with open(file_name, "wb") as file_pi:
            pickle.dump(mondict, file_pi)

    def get_pc(self, sequence, scan_ref, scan_in, curr_overlap):
        path_sequence = os.path.join(Param.path_sequence_base, sequence)
        ref_pc = "Hokuyo_" + str(scan_ref) + ".csv"
        if curr_overlap is None:
            in_pc = "Hokuyo_" + str(scan_in) + ".csv"
        else:
            in_pc = "Mo_" + str(scan_ref) + "_" + str(scan_in) + ".csv"
        ref_pc_path = os.path.join(path_sequence, Param.path_pc, ref_pc)
        in_pc_path = os.path.join(path_sequence, Param.path_pc, in_pc)
        return ref_pc_path, in_pc_path

    def get_mc_results(self, base_path, curr_overlap):
        # base_path = os.path.join(path, sequence, str(scan_ref))
        # path_p = os.path.join(base_path, 'mc.p')
        # if os.path.exists(path_p):
        #     mondict = self.load(path_p)
        #     return mondict['T_mc'], mondict['T_init_mc']

        n = 0
        T_mc = SE3.new(Param.n_mc)
        T_init_mc = SE3.new(Param.n_mc)

        # TODO: add scan_ref and scan_in to filename
        for n in range(Param.n_mc):
            if curr_overlap is None:
                filename = 'mc_' + str(n) + '.txt'
            else:
                filename = 'momc_' + str(n) + '.txt'
            path = os.path.join(base_path, filename)
            if not os.path.exists(path):
                break
            data = np.genfromtxt(path)
            T_mc[n] = data[:4]
            T_init_mc[n] = data[4:]
            # n += 1
        # T_mc = T_mc[:n]
        # T_init_mc = T_init_mc[:n]
        # mondict = {
        #     'T_mc': T_mc,
        #     'T_init_mc': T_init_mc
        #     }
        # self.dump(mondict, path_p)
        return T_mc, T_init_mc
