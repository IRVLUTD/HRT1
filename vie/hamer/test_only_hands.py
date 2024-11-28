import numpy as np
import open3d as o3d
import trimesh
import torch
from scipy.spatial.transform import Rotation as R

# Define paths
task_dir = '/home/jishnu/Projects/mm-demo/vie/imgs/test/000100'
right_hand_obj = f'{task_dir}/out/hamer/000100_1.obj'
left_hand_obj = f'{task_dir}/out/hamer/000100_0.obj'
hamer_out = f'{task_dir}/000100_hamer.pt'

# Load hand meshes from OBJ files
right_hand_mesh = trimesh.load_mesh(right_hand_obj, process=False)
left_hand_mesh = trimesh.load_mesh(left_hand_obj, process=False)

# Load hand transformation data
x = torch.load(hamer_out)
right_cam_rotation = x['pred_cam'][1].cpu().numpy()  # Rotation for right hand (3D vector)
right_cam_translation = x['pred_cam_t'][1].cpu().numpy()  # Translation for right hand (3,)

# Convert axis-angle rotation to a 3x3 rotation matrix for right hand
rotation_matrix = R.from_rotvec(right_cam_rotation).as_matrix()

# Construct the camera transformation matrix for right hand (4x4)
right_cam_transformation = np.eye(4)
right_cam_transformation[:3, :3] = rotation_matrix
right_cam_transformation[:3, 3] = right_cam_translation

# Load left hand transformation data
left_cam_rotation = x['pred_cam'][0].cpu().numpy()  # Rotation for left hand (3D vector)
left_cam_translation = x['pred_cam_t'][0].cpu().numpy()  # Translation for left hand (3,)

# Convert axis-angle rotation to a 3x3 rotation matrix for left hand
rotation_matrix_left = R.from_rotvec(left_cam_rotation).as_matrix()

# Construct the camera transformation matrix for left hand (4x4)
left_cam_transformation = np.eye(4)
left_cam_transformation[:3, :3] = rotation_matrix_left
left_cam_transformation[:3, 3] = left_cam_translation

# Align right hand mesh vertices to camera space
right_hand_vertices = np.array(right_hand_mesh.vertices)
right_hand_vertices_transformed = (rotation_matrix @ right_hand_vertices.T).T + right_cam_translation

# Align left hand mesh vertices to camera space
left_hand_vertices = np.array(left_hand_mesh.vertices)
left_hand_vertices_transformed = (rotation_matrix_left @ left_hand_vertices.T).T + left_cam_translation

# Visualize both hands with and without transformation to debug distance
# Convert to Open3D for visualization
# Right hand point cloud (before and after transformation)
right_hand_pcd = o3d.geometry.PointCloud()
right_hand_pcd.points = o3d.utility.Vector3dVector(right_hand_vertices)  # Untransformed
right_hand_pcd.paint_uniform_color([1, 0, 0])  # Red color

right_hand_pcd_transformed = o3d.geometry.PointCloud()
right_hand_pcd_transformed.points = o3d.utility.Vector3dVector(right_hand_vertices_transformed)  # Transformed
right_hand_pcd_transformed.paint_uniform_color([1, 0, 0])  # Red color

# Left hand point cloud (before and after transformation)
left_hand_pcd = o3d.geometry.PointCloud()
left_hand_pcd.points = o3d.utility.Vector3dVector(left_hand_vertices)  # Untransformed
left_hand_pcd.paint_uniform_color([0, 0, 1])  # Blue color

left_hand_pcd_transformed = o3d.geometry.PointCloud()
left_hand_pcd_transformed.points = o3d.utility.Vector3dVector(left_hand_vertices_transformed)  # Transformed
left_hand_pcd_transformed.paint_uniform_color([0, 0, 1])  # Blue color

# Visualize both transformed and untransformed hands to compare
o3d.visualization.draw_geometries([right_hand_pcd, left_hand_pcd,
                                   right_hand_pcd_transformed, left_hand_pcd_transformed],
                                  window_name="Untransformed and Transformed Hand Meshes")
