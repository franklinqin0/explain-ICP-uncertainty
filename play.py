import os
import numpy as np
import open3d as o3d
from overlap import calc_overlap
from dataset import Dataset

pc1_path = '/home/parallels/Desktop/idp/data/Apartment/local_frame/0_05/Hokuyo_1.csv'
pc1 = np.loadtxt(pc1_path, delimiter=",", skiprows=1)
# pc1 = pc1[:, 1:4]

pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
o3d.visualization.draw_geometries([pcd1],
                                zoom=0.3412,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

# pc2_path = '/home/parallels/Desktop/idp/data/Apartment/local_frame/0_05/Mo_1_2.csv'
pc2_path = '/home/parallels/Desktop/idp/data/Apartment/local_frame/0_05/Hokuyo_2.csv'
pc2 = np.loadtxt(pc2_path, delimiter=",", skiprows=1)

pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))
o3d.visualization.draw_geometries([pcd2],
                                zoom=0.3412,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

dataset = Dataset()
# overlap_mat = dataset.get_overlap_matrix(seq)
T_gt = dataset.get_data("Apartment")

ol = calc_overlap(pc1, pc2, T_gt[1], T_gt[2])
print(ol)
