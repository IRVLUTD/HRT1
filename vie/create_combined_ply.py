import numpy as np
import open3d as o3d

def merge_and_color_ply_files(ply_paths, output_path):
    """
    Merges multiple .ply files into one, applies specified colors, and applies
    both a 180-degree rotation and a reflection along the XY plane.
    
    Args:
        ply_paths (list of str): List of file paths to the .ply files.
        output_path (str): Path to save the merged .ply file.
    """
    merged_point_cloud = o3d.geometry.PointCloud()

    # Define colors (RGB values normalized between 0 and 1)
    orange_color = np.array([1.0, 1.0, 0.0])  # Orange
    red_color = np.array([1.0, 0.0, 0.0])     # Red

    for idx, ply_path in enumerate(ply_paths):
        if idx == 2:  # Third file (end-effector mesh)
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(ply_path)
            
            # Apply red color to the entire mesh
            mesh.paint_uniform_color(red_color)
            
            # Convert the mesh to a point cloud for merging
            mesh_point_cloud = mesh.sample_points_uniformly(number_of_points=5000)  # Adjust point density as needed
            merged_point_cloud += mesh_point_cloud
        else:
            # Load the point cloud
            point_cloud = o3d.io.read_point_cloud(ply_path)

            if idx == 0:
                # Keep original RGB for the first file
                if not point_cloud.has_colors():
                    print(f"Warning: First file {ply_path} has no colors. Using default white.")
            elif idx == 1:
                # Color the second file as orange
                point_cloud.colors = o3d.utility.Vector3dVector(
                    np.tile(orange_color, (np.asarray(point_cloud.points).shape[0], 1))
                )
            
            # Merge the current point cloud into the main one
            merged_point_cloud += point_cloud
    
    # Apply a 180-degree rotation along the y-axis
    print("Applying 180-degree rotation along the y-axis...")
    rotation_matrix = np.array([
        [-1, 0,  0, 0],  # Flip X
        [ 0, -1,  0, 0],  # Flip Y
        [ 0,  0, -1, 0],  # Flip Z
        [ 0,  0,  0, 1]
    ])
    merged_point_cloud.transform(rotation_matrix)

    # Apply a reflection along the XY plane
    print("Applying reflection along the XY plane...")
    reflection_matrix = np.array([
        [-1,  0,  0, 0],  # Keep X
        [0,  1,  0, 0],  # Keep Y
        [0,  0, 1, 0],  # Flip Z
        [0,  0,  0, 1]
    ])
    merged_point_cloud.transform(reflection_matrix)

    # Save the transformed point cloud to a new .ply file
    o3d.io.write_point_cloud(output_path, merged_point_cloud)
    print(f"Transformed (rotated and reflected) PLY file saved to {output_path}")

def process_task(root_dir, task_dir, hand_alias):
        _task_hamer_dir = f"{root_dir}/{task_dir}/out/hamer"
        output_dir = f"{_task_hamer_dir}/combined_ply"
        os.makedirs(output_dir, exist_ok=True)

        hand = hand_alias
        for f in os.listdir(f"{_task_hamer_dir}/scene"):
            try:
                name, ext = os.path.splitext(f)
                upd_f = f"{name}_{hand}{ext}"
                ply_files = [
                    f"{_task_hamer_dir}/scene/{f}",
                    f"{_task_hamer_dir}/3dhand/{upd_f}",
                    f"{_task_hamer_dir}/transfer_hand_mesh/{upd_f}"
                ]

                merge_and_color_ply_files(ply_files, f"{output_dir}/{f}")
            except:
                continue
    
# Example usage
import os
import sys

root_dir = sys.argv[1]

tasks_with_hand_alias = [
    ["", 1],
    ("task_20_microwave-open_interval_0.05", 1),
    ("task_12_12s-use-spatula", 0),
    ("task_16_12s-wipe-table-with-towel", 0),
    ("task_1_shelf-bottle_interval_0.05", 0),
    ("task_11_17s-use-basting-brush", 0),
    ("task_18_10s-move-chair", 0),
    ("task_14_15s-pouring", 1),
    ("task_10_15s-use-sponge-scrub", 1),
    ("task_3_14s-close-jar-with-lid", 0),
    ("task_22_water_fill_from_water_cooler", 0),
    ("task_21_whiteboard-eraser_interval_0.05", 0),
    ("task_13_13s-sprinkle-salt", 1),
    ("task_8_17s-use_hammer", 0),
    ("task_6_8s-press-keyboard-key", 0),
    ("task_5_15s-open-folder", 0),
    ("task_9-use-stapler", 1),
    ("task_4_17s-fold-towel", 0),
    ("task_19_fetch-shelf-ycb-red-mug_interval_0.05", 1),
    ("task_17_15s-squeeze-sponge-ball", 0),
    ("task_15_19s-use-knife", 0),
    ("task_7_14s-use-cleaning-brush", 0),
    ("task_2_9s-toggle-light-switch", 0),
]


for task_dir, hand_alias in tasks_with_hand_alias:
    process_task(root_dir, task_dir, hand_alias)
    print(task_dir)
    # break
    # input(f'Finished {task_dir}. Go to next?')
