#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import open3d as o3d

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    """
    Converts depth image to 3D point cloud (xyz coordinates).
    
    Args:
    - depth_img: Depth image in meters.
    - fx, fy: Focal lengths in pixels.
    - px, py: Principal point coordinates in pixels.
    - height, width: Dimensions of the depth image.
    
    Returns:
    - xyz_img: 3D point cloud in world coordinates.
    """
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img

def load_rgb_image(image_path):
    """
    Loads the RGB image from the specified path.
    
    Args:
    - image_path: Path to the RGB image.
    
    Returns:
    - rgb_image: The loaded RGB image.
    """
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise ValueError(f"Error: Unable to load RGB image from {image_path}")
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

def project_point_to_pixel(x, y, z, fx, fy, px, py):
    """
    Projects a 3D point (x, y, z) into 2D pixel coordinates (u, v) using the camera's intrinsic parameters.
    
    Args:
    - x, y, z: 3D coordinates in the camera frame.
    - fx, fy: Focal lengths (in pixels) in the x and y axes.
    - px, py: Principal point coordinates (usually the optical center) in pixels.

    Returns:
    - u, v: 2D pixel coordinates in the image plane.
    """
    u = (fx * x / z) + px  # Projection onto the u (x-axis) pixel coordinate
    v = (fy * y / z) + py  # Projection onto the v (y-axis) pixel coordinate
    return int(u), int(v)

def save_point_cloud_with_intensity(pcd, output_ply_path):
    """
    Save the point cloud with RGB and intensity attributes to a PLY file manually.
    
    Args:
    - pcd: Point cloud object.
    - output_ply_path: Path to the output PLY file.
    """
    # Manually write PLY file including the intensity attribute
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    intensities = np.ones(len(points))  # Intensity set to 1 for all points

    # Open file for writing
    with open(output_ply_path, 'w') as f:
        # Write PLY header
        f.write(f"ply\n")
        f.write(f"format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write(f"property float x\n")
        f.write(f"property float y\n")
        f.write(f"property float z\n")
        f.write(f"property float intensity\n")  # Add intensity property between xyz and rgb
        f.write(f"property uchar red\n")
        f.write(f"property uchar green\n")
        f.write(f"property uchar blue\n")
        f.write(f"end_header\n")

        # Write point cloud data
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            intensity = intensities[i]
            f.write(f"{x} {y} {z} {intensity} {int(r * 255)} {int(g * 255)} {int(b * 255)}\n")

    print(f"Point cloud with intensity saved as {output_ply_path}")

def visualize_and_save_point_cloud_with_rgb_and_intensity(depth_path, rgb_image_path, fx, fy, px, py, output_ply_path):
    """
    Computes the point cloud from the depth image, maps RGB colors, stores intensity (1), and saves it as a PLY file.
    
    Args:
    - depth_path: Path to the depth image.
    - rgb_image_path: Path to the RGB image.
    - fx, fy, px, py: Camera intrinsic parameters.
    - output_ply_path: Path to save the output PLY file.
    """
    # Check if depth image exists
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found at {depth_path}")
    
    # Load depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ValueError(f"Failed to load depth image from {depth_path}")
    
    depth_img = depth_img / 1000.0  # Convert depth to meters

    # Load RGB image
    rgb_image = load_rgb_image(rgb_image_path)

    # Get image height and width
    height, width = depth_img.shape

    # Compute 3D points from depth
    xyz_img = compute_xyz(depth_img, fx, fy, px, py, height, width)

    # Initialize the color array for the point cloud
    colors = np.zeros((height, width, 3), dtype=np.float32)

    # Project 3D points to 2D pixels and get RGB values
    for v in range(height):
        for u in range(width):
            x, y, z = xyz_img[v, u]
            if z == 0:  # Skip invalid points (zero depth)
                continue
            # Project to 2D pixel coordinates
            pixel_u, pixel_v = project_point_to_pixel(x, y, z, fx, fy, px, py)

            # Ensure the projected 2D pixel is within image bounds
            if 0 <= pixel_u < width and 0 <= pixel_v < height:
                r, g, b = rgb_image[pixel_v, pixel_u]
                colors[v, u] = [r / 255.0, g / 255.0, b / 255.0]  # Normalize RGB

    # Create point cloud from XYZ points
    points = xyz_img.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)

    # Visualize the point cloud with RGB colors
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with RGB Overlay")

    # Save the point cloud as a PLY file (including intensity)
    save_point_cloud_with_intensity(pcd, output_ply_path)
    print(f"Point cloud saved as {output_ply_path}")

# Example usage:
depth_path = "../_DATA/human_demonstrations/whiteboard-eraser_interval_0.05/depth/000000_depth.png"
rgb_image_path = "../_DATA/human_demonstrations/whiteboard-eraser_interval_0.05/rgb/000000_color.png"
output_ply_path = "output_point_cloud_with_intensity.ply"

visualize_and_save_point_cloud_with_rgb_and_intensity(depth_path, rgb_image_path, fx=574.0527954101562, fy=574.0527954101562, px=319.5, py=239.5, output_ply_path=output_ply_path)
