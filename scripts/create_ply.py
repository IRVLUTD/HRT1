#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import numpy as np
import cv2
import os

def compute_xyz(depth_img, fx, fy, px, py):
    height, width = depth_img.shape
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img

def save_ply(xyz_img, rgb_img, filename="output.ply"):
    height, width, _ = xyz_img.shape
    vertices = []
    
    for i in range(height):
        for j in range(width):
            x, y, z = xyz_img[i, j]
            if z > 0:  # Ignore points with zero depth
                r, g, b = rgb_img[i, j]
                vertices.append(f"{x} {y} {z} {r} {g} {b}")
    
    ply_header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as ply_file:
        ply_file.write(ply_header + "\n".join(vertices))

# Load the images
rgb_path = "../_DATA/human_demonstrations/whiteboard-eraser_interval_0.05/rgb/000000_color.png"
depth_path = "../_DATA/human_demonstrations/whiteboard-eraser_interval_0.05/depth/000000_depth.png"

rgb_img = cv2.imread(rgb_path)
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # Convert depth to meters if needed

# Camera intrinsics
intrinsics = [
    [574.0527954101562, 0.0, 319.5],
    [0.0, 574.0527954101562, 239.5],
    [0.0, 0.0, 1.0],
]
fx, fy, px, py = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]

# Compute XYZ
xyz_img = compute_xyz(depth_img, fx, fy, px, py)

# Save to PLY
save_ply(xyz_img, rgb_img, filename="output.ply")

print("PLY file saved as output.ply")
