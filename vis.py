import numpy as np
import open3d as o3d

# Load the point cloud data from a CSV file
pcd = np.loadtxt("/home/parallels/Desktop/haha/data/Apartment/lf_sensor/Hokuyo_14.csv", delimiter=",", skiprows=1)

# add sensor noise
pcd_arr = pcd[:, 1:4] # np.asarray(pcd.points)
# noise = np.random.normal(0, 0.02, pcd_arr.shape)
# pcd_arr += noise

# convert the numpy array to an Open3D point cloud object
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_arr))

# Sample num_points points from the point cloud
# pcd = pcd.uniform_down_sample(every_k_points=len(pcd.points) // 2048)

# or fps?

# print(np.asarray(pcd.points)[:10])

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


