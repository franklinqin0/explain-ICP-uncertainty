import os
import numpy as np
import pickle
from utils import *

class Dataset:
    path_sequence_base = '/Users/ziyuan/Desktop/Github/idp'
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

    def __init__(self):
        self.read_data()

    def read_data(self):
        for sequence in self.sequences:
            self.preprocessing_data(sequence)

    def preprocessing_data(self, sequence):
        path_sequence = os.path.join(self.path_sequence_base, sequence)
        path_pickle = os.path.join(path_sequence, 'data.p')
        # if not Param.b_data and os.path.exists(path_pickle):
        #     print('Data already preprocessed')
        # else:
        files = os.listdir(os.path.join(path_sequence, 'local_frame'))
        n_scan = int((len(files)-1)/4)
        T_gt_file = os.path.join(path_sequence, 'global_frame', 'pose_scanner_leica.csv')
        T_gt_data = np.genfromtxt(T_gt_file, delimiter=',', skip_header=1)
        T_gt = SE3.new(n_scan)

        for k in range(3):
            T_gt[:, k, :] = T_gt_data[:, 2+4*k:6+4*k]
        mondict = {
            'T_gt': T_gt,
        }
        self.dump(mondict, path_pickle)

    def get_data(self, sequence):
        path_sequence = os.path.join(self.path_sequence_base, sequence)
        path = os.path.join(path_sequence, 'data.p')
        mondict = self.load(path)
        return mondict['T_gt']

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

    def get_pc(self, sequence, k):
        path_sequence = os.path.join(self.path_sequence_base, sequence)
        pc_file = "Hokuyo_" + str(k) + ".csv"
        return os.path.join(path_sequence, 'local_frame', pc_file)

    def get_mc_results(self, sequence, scan_ref):
        base_path = os.path.join('results', sequence, str(scan_ref))
        path_p = os.path.join(base_path, 'mc.p')
        # if os.path.exists(path_p):
        #     mondict = self.load(path_p)
        #     return mondict['T_mc'], mondict['T_init_mc']

        n = 0
        T_mc = SE3.new(Param.n_mc)
        T_init_mc = SE3.new(Param.n_mc)
        while True:
            path = os.path.join(base_path, 'mc_' + str(n) + '.txt')
            if not os.path.exists(path):
                break
            data = np.genfromtxt(path)
            T_mc[n] = data[:4]
            T_init_mc[n] = data[4:]
            n += 1
        T_mc = T_mc[:n]
        T_init_mc = T_init_mc[:n]
        mondict = {
            'T_mc': T_mc,
            'T_init_mc': T_init_mc
            }
        self.dump(mondict, path_p)
        # print("scan_ref: ", scan_ref)
        # print(T_mc)
        # print('---')
        # print(T_init_mc)
        return T_mc, T_init_mc
