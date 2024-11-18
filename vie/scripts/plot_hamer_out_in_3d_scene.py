import torch
import numpy as np
import open3d as o3d
from PIL import Image

def depth_to_pointcloud(rgb, depth, intrinsics):
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    
    x = np.tile(np.arange(w), (h, 1))
    y = np.tile(np.arange(h).reshape(-1, 1), (1, w))
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    return points, colors

def apply_transformation(points, RT_camera):
    R = RT_camera[:3, :3]
    t = RT_camera[:3, 3]
    if points.shape[1] == 3:
        points_world = (R @ points.T).T + t
    else:
        raise ValueError(f"Points have an unexpected shape: {points.shape}")
    return points_world

def plot_hand_keypoints_on_pointcloud(hand_keypoints):
    """
    Create and return a list of spheres representing hand keypoints.
    """
    spheres = []
    for keypoint in hand_keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(keypoint)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for the hand keypoints
        spheres.append(sphere)
    return spheres

def add_hand_mesh_to_pointcloud(hand_vertices, hand_faces=None, color=[1.0, 0.0, 0.0]):
    """
    Add hand mesh to the Open3D point cloud visualization.
    """
    hand_mesh = o3d.geometry.TriangleMesh()
    
    # Ensure hand_vertices is a numpy array of shape (n, 3)
    hand_vertices = np.array(hand_vertices)
    

    print(hand_vertices.shape)


    import pdb; pdb.set_trace()

    # Create Vector3dVector for vertices
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_vertices)
    
    # If faces are provided, assign them to the hand mesh
    if hand_faces is not None:
        hand_faces = np.array(hand_faces)
        hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    
    # Apply color to the mesh
    hand_mesh.paint_uniform_color(color)
    
    hand_pcd = hand_mesh.sample_points_uniformly(number_of_points=1000)
    return hand_pcd

if __name__ == "__main__":
    # File paths
    root_dir = "../imgs/test/000100"
    rgb_path = f"{root_dir}/rgb/000100.jpg"
    depth_path = f"{root_dir}/depth/000100_depth.png"
    pose_path = f"{root_dir}/pose/000100_pose.npz"
    hamer_out_path = f"{root_dir}/000100_hamer.pt"
    
    # Load data
    rgb = np.asarray(Image.open(rgb_path))
    depth = np.asarray(Image.open(depth_path)) / 1000.0  # Convert depth to meters
    pose = np.load(pose_path)
    RT_camera = pose['RT_camera']
    
    # Camera intrinsics
    intrinsics = {
        'fx': 525.0,  # Replace with your camera's focal length in x
        'fy': 525.0,  # Replace with your camera's focal length in y
        'cx': rgb.shape[1] / 2,
        'cy': rgb.shape[0] / 2
    }
    
    # Generate point cloud
    points, colors = depth_to_pointcloud(rgb, depth, intrinsics)
    
    # Transform point cloud to world coordinates
    points_world = apply_transformation(points, RT_camera)
    
    # Load HAMER model output (hand mesh)
    hamer_output = torch.load(hamer_out_path)
    hand_vertices = hamer_output['pred_vertices'].cpu().numpy()  # Extract vertices
    hand_keypoints_3d = hamer_output['pred_keypoints_3d'].cpu().numpy()  # Extract 3D keypoints
    
    # Transform hand keypoints to world coordinates
    hand_keypoints_world = apply_transformation(hand_keypoints_3d.reshape(-1, 3), RT_camera)
    
    # Visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Plot hand keypoints
    spheres = plot_hand_keypoints_on_pointcloud(hand_keypoints_world)
    
    # Add hand mesh to point cloud (optional)
    hand_pcd = add_hand_mesh_to_pointcloud(hand_vertices)
    
    # Show combined visualization
    o3d.visualization.draw_geometries([pcd] + spheres + [hand_pcd])
