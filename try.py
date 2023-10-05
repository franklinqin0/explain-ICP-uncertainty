from utils import *
from overlap import *
from dataset import Dataset

# Param.sensor_noise = 0.01
# Param.init_unc = 1.0
# Param.update()
# seq = "Apartment"
# Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0))
# Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc))
# print("results_path:", Param.results_path)
# print("results_pert:", Param.results_pert)
# print("path_pc:", Param.path_pc)

import os
import numpy as np
from pykdtree.kdtree import KDTree
import open3d as o3d

overlap_mat_path = "overlap_apartment.csv"
overlap_mat = np.genfromtxt(overlap_mat_path, delimiter=',')



if __name__ == "__main__":
    local_frame = "data/Apartment/local_frame/0_0"

    i = 1
    j = 5

    pc1_path = os.path.join(local_frame, f"Hokuyo_{i}.csv")
    pc1 = np.loadtxt(pc1_path, delimiter=",", skiprows=1)
    pc1 = pc1[:, 1:4]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
    
    pc2_path = os.path.join(local_frame, f"Hokuyo_{j}.csv")
    pc2 = np.loadtxt(pc2_path, delimiter=",", skiprows=1)
    pc2 = pc2[:, 1:4]
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

    print("current ratio:", overlap_mat[i, j])

    dataset = Dataset()
    sequence = "Apartment"
    T_gt = dataset.get_data(sequence)

    t1 = T_gt[i]
    t2 = T_gt[j]
    pc2_updated = adjust_overlap(pc1, pc2, t1, t2, overlap_mat[i, j], 0.9)
    updated_ratio = calc_overlap(pc1, pc2_updated, t1, t2)
    print("updated ratio", updated_ratio)
    
    # convert the numpy array to an Open3D point cloud object
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2_updated))

    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])



#####

    lpc2_path = os.path.join(local_frame, f"Hokuyo_{j}.csv")
    lpc2 = np.loadtxt(lpc2_path, delimiter=",", skiprows=1)

    # if 
    
    
    # adjust the `scan_in` point cloud to reach overlap ratio `po`
    if abs(po - Param.mean_overlap_ratio) < 1e-3:
        filename = "Pom_{}" + str(scan_ref) + ".csv"
        curr_ratio = calc_overlap()
        adjust_overlap(clean_input_folder, pert_input_folder, scan_ref, T_gt[scan_ref], T_gt[scan_in], curr_ratio, po)
    else:
        filename = "Hokuyo_" + str(scan_ref) + ".csv"
    
    
    
    filepath = os.path.join(clean_input_folder)
    
    
    
    
        # df_ref['x'] += np.random.normal(0, noise_stddev, df_ref.shape[0])
        # df_ref['y'] += np.random.normal(0, noise_stddev, df_ref.shape[0])
        # df_ref['z'] += np.random.normal(0, noise_stddev, df_ref.shape[0])
        # df_ref = df_ref[['x', 'y', 'z']]
        # df_ref.to_csv(os.path.join(output_folder, ref_filename), index=False)
        
        # df_in['x'] += np.random.normal(0, noise_stddev, df_in.shape[0])
        # df_in['y'] += np.random.normal(0, noise_stddev, df_in.shape[0])
        # df_in['z'] += np.random.normal(0, noise_stddev, df_in.shape[0])
        # df_in = df_in[['x', 'y', 'z']]
        # df_in.to_csv(os.path.join(output_folder, in_filename), index=False)