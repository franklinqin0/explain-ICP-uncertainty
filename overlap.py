import os
import numpy as np
from pykdtree.kdtree import KDTree
import open3d as o3d
from utils import *
from dataset import Dataset

def calc_overlap(pc1_l, pc2_l, t1, t2, distance=0.2):
    pc1 = transform_point_cloud(pc1_l, t1)
    pc2 = transform_point_cloud(pc2_l, t2)
    
    cloud1_tree = KDTree(pc1)
    dist, idx = cloud1_tree.query(pc2, 1, eps=distance/100, distance_upper_bound=distance)
    neighbors_found = np.count_nonzero(np.isfinite(dist))
    overlap_ratio = neighbors_found / len(pc1_l)
    return overlap_ratio


def reduce_overlap(pc1_l, pc2, t1, t2, target_ratio, distance=0.2):
    """
    Reduce the overlap ratio by reducing overlapping region of point cloud 2
    with respect to point cloud 1.
    """
    
    pc1 = transform_point_cloud(pc1_l, SE3.mul(SE3.inv(t2), t1))
    # pc2_g = transform_point_cloud(pc2_l, t2)
    
    cloud1_tree = KDTree(pc1)
    dist, idx = cloud1_tree.query(pc2, 1, eps=distance/100, distance_upper_bound=distance)
    
    # get overlapping points' indices from pc2
    overlapping_indices = np.where(np.isfinite(dist))[0]
    
    # calculate the number of overlapping points to remove to achieve target_ratio
    points_to_remove = round(len(overlapping_indices) - len(pc1) * target_ratio)
    
    # randomly select points to remove from overlapping_indices
    np.random.seed(4)
    indices_to_remove = np.random.choice(overlapping_indices, size=points_to_remove, replace=False)
    
    # remove these indices from pc2
    reduced_pc2 = np.delete(pc2, indices_to_remove, axis=0)
    
    return reduced_pc2


def increase_overlap(pc1_l, pc2, t1, t2, target_ratio, distance=0.2, perturbation_factor=0.001):
    # pc1_g = transform_point_cloud(pc1_l)
    # pc2 = transform_point_cloud(pc2_l, SE3.mul(SE3.inv(t1), t2))
    pc1 = transform_point_cloud(pc1_l, SE3.mul(SE3.inv(t2), t1))
    
    cloud1_tree = KDTree(pc1)
    cloud2_tree = KDTree(pc2)
    
    # distances and indices of the nearest neighbors in cloud1 for each point in cloud2
    dist1, idx1 = cloud1_tree.query(pc2, 1, eps=distance/100, distance_upper_bound=distance)
    
    # distances and indices of the nearest neighbors in cloud2 for each point in cloud1
    dist2, idx2 = cloud2_tree.query(pc1, 1, eps=distance/100, distance_upper_bound=distance)
    
    # get points in cloud1 that are not overlapping with cloud2
    non_overlapping_indices = np.where(np.isinf(dist2))[0]
    
    overlapping_indices = np.where(np.isfinite(dist1))[0]
    
    # calculate the number of non-overlapping points to add to achieve target_ratio
    points_to_add = round(len(pc1) * target_ratio - len(overlapping_indices))
    
    # randomly select 'points_to_add' points from non_overlapping_indices
    np.random.seed(4)
    indices_to_add = np.random.choice(non_overlapping_indices, size=min(points_to_add, len(non_overlapping_indices)), replace=False)
    
    if points_to_add > len(non_overlapping_indices):
        print(points_to_add, len(non_overlapping_indices))

    # add these points from pc1 to pc2 with slight perturbation
    perturbation = np.random.normal(loc=0, scale=perturbation_factor, size=pc1[indices_to_add].shape)
    increased_pc2 = np.vstack([pc2, pc1[indices_to_add] + perturbation])
    
    return increased_pc2


def adjust_overlap(pc1, pc2, t1, t2, target_ratio):
    """
    Adjust point cloud 2 to achieve the target ratio.
    """
    current_ratio = calc_overlap(pc1, pc2, t1, t2)
    if abs(current_ratio - target_ratio) < 1e-3:
        return pc2
    elif current_ratio < target_ratio:
        return increase_overlap(pc1, pc2, t1, t2, target_ratio)
    elif current_ratio > target_ratio:
        return reduce_overlap(pc1, pc2, t1, t2, target_ratio)


def add_noise(input_folder, output_folder, scan_ref, scan_in, noise_stddev, T_gt, curr_overlap):
    """
    Add noise to scan_ref output and scan_in output point cloud files.
    If `curr_overlap` is None, adjust scan_in output point cloud to satisfy
    mean overlap ratio.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ref_input_filename = "Hokuyo_" + str(scan_ref) + ".csv"
    in_input_filename = "Hokuyo_" + str(scan_in) + ".csv"
    ref_input_path = os.path.join(input_folder, ref_input_filename)
    in_input_path = os.path.join(input_folder, in_input_filename)
    
    ref_output_path = os.path.join(output_folder, ref_input_filename)
    
    if curr_overlap is not None:
        # ref_output_filename = "Mo_" + str(scan_ref) + "_" + str(scan_in) + ".csv"
        in_output_filename = "Mo_" + str(scan_ref) + "_" + str(scan_in) + ".csv"
        in_output_path = os.path.join(output_folder, in_output_filename)
        
        if os.path.exists(ref_output_path) and os.path.exists(in_output_path):
            return

        df_ref = pd.read_csv(ref_input_path)
        pc_ref = df_ref[['x', 'y', 'z']].values
        df_in = pd.read_csv(in_input_path)
        pc_in = df_in[['x', 'y', 'z']].values
        
        # add noise before adjusting overlap
        pc_ref += np.random.normal(0, noise_stddev, pc_ref.shape)
        pc_in += np.random.normal(0, noise_stddev, pc_in.shape)
        pc_in = adjust_overlap(pc_ref, pc_in, T_gt[scan_ref], T_gt[scan_in], Param.mean_overlap)
        
        if not os.path.exists(ref_output_path):
            np.savetxt(ref_output_path, pc_ref, delimiter=',', header="x,y,z", comments="")
        np.savetxt(in_output_path, pc_in, delimiter=",", header="x,y,z", comments="")
    else:
        in_output_path = os.path.join(output_folder, in_input_filename)
        
        if os.path.exists(ref_output_path) and os.path.exists(in_output_path):
            return

        df_ref = pd.read_csv(ref_input_path)
        pc_ref = df_ref[['x', 'y', 'z']].values
        df_in = pd.read_csv(in_input_path)
        pc_in = df_in[['x', 'y', 'z']].values
        
        pc_in += np.random.normal(0, noise_stddev, pc_in.shape)
        
        if not os.path.exists(ref_output_path):
            pc_ref += np.random.normal(0, noise_stddev, pc_ref.shape)
            np.savetxt(ref_output_path, pc_ref, delimiter=',', header="x,y,z", comments="")
        np.savetxt(in_output_path, pc_in, delimiter=',', header="x,y,z", comments="")

    
def mean_overlap(r, c, overlap_mat):
    diff = c - r
    lst = []

    for i in range(overlap_mat.shape[0]):
        j = i + diff
        if (j < 0) or (j > overlap_mat.shape[1]-1):
            break

        # print('i:', i, 'j:', j, 'val:', overlap_mat[i, j])
        lst.append(overlap_mat[i, j])

    return np.mean(np.asarray(lst))


def save_overlap():
    global_frame = "data/Apartment/global_frame"
    overlap_mat_path = os.path.join(global_frame, "overlap_apartment.csv")
    overlap_mat = np.genfromtxt(overlap_mat_path, delimiter=',')
    overlap_mat = overlap_mat[:, :-1] # rid last column, which is all None's

    res = np.ones_like(overlap_mat)
    for i in range(overlap_mat.shape[0]):
        for j in range(overlap_mat.shape[1]):
            if i != j:
                print("i:", i, "j:", j)
                pc1_path = os.path.join(global_frame, f"PointCloud{i}.csv")
                pc1 = np.loadtxt(pc1_path, delimiter=",", skiprows=1)
                pc1 = pc1[:, 1:4]
                
                pc2_path = os.path.join(global_frame, f"PointCloud{j}.csv")
                pc2 = np.loadtxt(pc2_path, delimiter=",", skiprows=1)
                pc2 = pc2[:, 1:4]
                
                res[i, j] = calc_overlap(pc1, pc2)

    diff = np.abs(res - overlap_mat)
    print(np.sum(diff))

    np.savetxt("overlap_apartment.csv", res, delimiter=",")


def transform_point_cloud(cloud, transform):
    """Transforms a point cloud using a transformation matrix."""
    # convert to homogeneous coordinates
    homogeneous_coords = np.hstack([cloud, np.ones((cloud.shape[0], 1))])
    transformed_cloud = np.dot(transform, homogeneous_coords.T).T
    return transformed_cloud[:, :3]  # convert back to 3D points


"""
i, j = 0, 4
global_frame = "data/Apartment/global_frame"
local_frame = "data/Apartment/local_frame/0_0"

pc1_path = os.path.join(global_frame, f"PointCloud{i}.csv")
pc1 = np.loadtxt(pc1_path, delimiter=",", skiprows=1)
pc1 = pc1[:, 1:4]

pc2_path = os.path.join(global_frame, f"PointCloud{j}.csv")
pc2 = np.loadtxt(pc2_path, delimiter=",", skiprows=1)
pc2 = pc2[:, 1:4]

dataset = Dataset()
sequence = "Apartment"
T_gt = dataset.get_data(sequence)

cloud2_local = transform_point_cloud(pc2, SE3.inv(T_gt[j]))

# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud2_local))

# o3d.visualization.draw_geometries([pcd],
#                                     zoom=0.3412,
#                                     front=[0.4257, -0.2125, -0.8795],
#                                     lookat=[2.6172, 2.0475, 1.532],
#                                     up=[-0.0694, -0.9768, 0.2024])

lpc2_path = os.path.join(local_frame, f"Hokuyo_{j}.csv")
lpc2 = np.loadtxt(lpc2_path, delimiter=",", skiprows=1)
lpc2 = lpc2[:, 1:4]

gpc2 = transform_point_cloud(lpc2, SE3.mul(SE3.inv(T_gt[i]), T_gt[j]))
comb = np.vstack([gpc2, pc1])

lpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(comb))

o3d.visualization.draw_geometries([lpcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

# for i in range(100):
#     print(cloud2_local[i], lpc2[i])
"""
