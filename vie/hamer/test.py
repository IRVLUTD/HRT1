import numpy as np
import open3d as o3d
import torch

# Define paths and load the model (replace with your actual data)
task_dir = '/home/jishnu/Projects/mm-demo/vie/imgs/test/000100'
hamer_out = f'{task_dir}/000100_hamer.pt'
pose = np.load(f'{task_dir}/pose/000100.npz')
hamer_out = f'{task_dir}/000100_hamer.pt'

cam_RT = pose['RT_camera']
cam_K = np.array([
    [574.0527954101562, 0.0, 319.5],
    [0.0, 574.0527954101562, 239.5],
    [0.0, 0.0, 1.0],
])

H = torch.load(hamer_out)
# (Pdb) h_out.keys()
# dict_keys(['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d'])
# dict_keys(['global_orient', 'hand_pose', 'betas'])

# h_out['pred_mano_params']




# Example dummy 3D hand mesh vertices (Replace with actual MANO model output)
X_world = H['pred_vertices'].cpu()

print(type(X_world))

# import pdb; pdb.set_trace()
# Example point cloud (replace with real data)
point_cloud = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])

# Convert to torch tensors
X_world = torch.tensor(X_world, dtype=torch.float32)
point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

# Define the camera parameters
K = torch.tensor(cam_K, dtype=torch.float32).unsqueeze(0)  # 3x3 intrinsic matrix

# Step 1: Apply camera extrinsic transformation (RT) to the world coordinates
R = torch.tensor(cam_RT[:3, :3], dtype=torch.float32)
t = torch.tensor(cam_RT[:3, 3], dtype=torch.float32)

# Transform the world points to camera space
X_camera = torch.matmul(R, X_world.T).T + t

# Step 2: Visualize using Open3D

# Create the point cloud for the 3D hand mesh
pcd_mesh = o3d.geometry.PointCloud()
pcd_mesh.points = o3d.utility.Vector3dVector(X_camera.numpy())

# Create a separate point cloud for visualization
pcd_points = o3d.geometry.PointCloud()
pcd_points.points = o3d.utility.Vector3dVector(point_cloud.numpy())

# Color the mesh (hand mesh) green
pcd_mesh.paint_uniform_color([0, 1, 0])  # Green color for hand mesh

# Color the point cloud red
pcd_points.paint_uniform_color([1, 0, 0])  # Red color for point cloud

# Step 3: Visualize the results
# Create a visualizer and add the point clouds (mesh and points)
o3d.visualization.draw_geometries([pcd_mesh, pcd_points],
                                  window_name="3D Hand Mesh and Point Cloud",
                                  width=800, height=600)

