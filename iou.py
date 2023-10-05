import open3d as o3d
import numpy as np

# Load the point cloud data from a CSV file
pcd0 = np.loadtxt("data/Apartment/global_frame/PointCloud0.csv", delimiter=",", skiprows=1)
pcd_arr0 = pcd0[:, 1:4]
pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_arr0))

pcd1 = np.loadtxt("data/Apartment/global_frame/PointCloud1.csv", delimiter=",", skiprows=1)
pcd_arr1 = pcd1[:, 1:4]
pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_arr1))

def compute_iou_from_point_clouds(point_cloud_1, point_cloud_2, voxel_size):
    """
    Compute IoU between two point clouds.
    
    Args:
    - point_cloud_1, point_cloud_2 (o3d.geometry.PointCloud): Open3D point cloud objects.
    - voxel_size (float): Size of each voxel.
    
    Returns:
    - IoU (float)
    """
    # Convert point clouds to voxel grids
    voxel_grid_1 = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_1, voxel_size=voxel_size)
    voxel_grid_2 = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_2, voxel_size=voxel_size)
    
    # Convert voxel grids to sets for fast computation of intersection and union
    voxel_set_1 = set([tuple(voxel.grid_index) for voxel in voxel_grid_1.get_voxels()])
    voxel_set_2 = set([tuple(voxel.grid_index) for voxel in voxel_grid_2.get_voxels()])
    
    # Compute intersection and union
    intersection = voxel_set_1.intersection(voxel_set_2)
    union = voxel_set_1.union(voxel_set_2)
    
    # Compute IoU
    iou = len(intersection) / len(union)
    return iou

# Load or create example point clouds
# Here, we create them from random data, but in practice, you'd likely load them from files.
# point_cloud_1 = o3d.geometry.PointCloud()
# point_cloud_1.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# point_cloud_2 = o3d.geometry.PointCloud()
# point_cloud_2.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# Compute IoU
voxel_size = 3
iou = compute_iou_from_point_clouds(pcd0, pcd1, voxel_size)
print(f"Overlap Ratio (IoU): {iou:.2f}")
