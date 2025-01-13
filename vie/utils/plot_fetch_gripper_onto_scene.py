import os
import open3d as o3d
import numpy as np

# Define paths
root_dir = "./_DATA/human_demonstrations/fetch-shelf-ycb-red-mug_interval_0.05/out/hamer/"
scene_dir = os.path.join(root_dir, "scene")
transfer_hand_mesh_dir = os.path.join(root_dir, "transfer_hand_mesh")

# Read the main scene file
scene_file = os.path.join(scene_dir, "000000.ply")

# Check if scene file is a point cloud or mesh
scene_data = o3d.io.read_triangle_mesh(scene_file)
if len(scene_data.triangles) == 0:
    # If no triangles, assume it's a point cloud
    print("Scene file is a point cloud. Loading as PointCloud...")
    scene_point_cloud = o3d.io.read_point_cloud(scene_file)
else:
    # If it's a triangular mesh, sample points
    print("Scene file is a mesh. Converting to PointCloud...")
    scene_point_cloud = scene_data.sample_points_uniformly(number_of_points=50000)

# Get all files matching *_1.ply and sort them by the number before '_1.ply'
files = [
    filename for filename in os.listdir(transfer_hand_mesh_dir) if filename.endswith("_1.ply")
]
files.sort(key=lambda x: int(x.split("_")[0]))

# Use a step size of 4
files = files[::4]

# Number of files for color interpolation
n_files = len(files)

# Combine with all files in transfer_hand_mesh/ and apply color gradient
combined_cloud = scene_point_cloud  # Start with the scene point cloud

for idx, filename in enumerate(files):
    file_path = os.path.join(transfer_hand_mesh_dir, filename)
    transfer_mesh = o3d.io.read_triangle_mesh(file_path)

    # Convert transfer mesh to a point cloud
    transfer_point_cloud = transfer_mesh.sample_points_uniformly(number_of_points=5000)  # Adjust as needed

    # Calculate color gradient (red to yellow)
    color = np.array([1, idx / n_files, 0])  # Transition from red [1, 0, 0] to yellow [1, 1, 0]

    # Apply the color to all points in the transfer_point_cloud
    colors = np.tile(color, (np.asarray(transfer_point_cloud.points).shape[0], 1))
    transfer_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Combine the transfer point cloud with the final point cloud
    combined_cloud += transfer_point_cloud

# Save the combined point cloud
output_file = os.path.join(root_dir, "all_gripper_and_scene_step10.ply")
o3d.io.write_point_cloud(output_file, combined_cloud)
print(f"Combined point cloud with step size of 10 saved to {output_file}")
