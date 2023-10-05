import os
import numpy as np
from overlap import overlap


# TODO: waiting for answer
global_frame = "data/Apartment/global_frame"
overlap_mat_path = os.path.join(global_frame, "overlap_apartment.csv")
overlap_mat = np.genfromtxt(overlap_mat_path, delimiter=',')
overlap_mat = overlap_mat[:, :-1] # rid last column, which is all None's

# print('given:', overlap_mat[0])

for j in range(overlap_mat.shape[0]):
    pc1_path = os.path.join(global_frame, f"PointCloud0.csv")
    pc1 = np.loadtxt(pc1_path, delimiter=",", skiprows=1)
    pc1 = pc1[:, 1:4]
    
    pc2_path = os.path.join(global_frame, f"PointCloud{j}.csv")
    pc2 = np.loadtxt(pc2_path, delimiter=",", skiprows=1)
    pc2 = pc2[:, 1:4]
    print(overlap_mat[0, j], overlap(pc1, pc2, distance=0.2))
