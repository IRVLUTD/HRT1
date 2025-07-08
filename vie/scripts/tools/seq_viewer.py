import open3d as o3d
import numpy as np
import os
import argparse
from open3d.visualization import gui
from pathlib import Path

# Try to import screeninfo for automatic resolution detection
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False

def get_screen_resolution():
    """Get the primary monitor's resolution, or return default if unavailable."""
    if SCREENINFO_AVAILABLE:
        try:
            # Get primary monitor
            for monitor in get_monitors():
                if monitor.is_primary:
                    return monitor.width, monitor.height
            # If no primary monitor, use first monitor
            monitor = get_monitors()[0]
            return monitor.width, monitor.height
        except Exception:
            pass
    # Default resolution if screeninfo fails or is unavailable
    return 1920, 1080

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/ply_sequence/', help='Base directory containing hamer/scene, hamer/3dhand, and hamer/transfer_hand_mesh')
    parser.add_argument('--num_points', type=int, default=10000000000, help='Number of points to sample from each point cloud')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for auto mode')
    parser.add_argument('--skip_viz_frames', type=int, default=1, help='Show every Nth frame in auto mode or skip N frames per key press in manual mode (1 for all frames)')
    parser.add_argument('--auto_mode', action='store_true', default=False, help='Use auto mode with cyclic playback (default: False, i.e., manual mode)')
    parser.add_argument('--left_hand', action='store_true', default=False, help='Use left hand files with _0 suffix (default: False, i.e., right hand with _1 suffix)')
    return parser.parse_args()

def merge_and_color_ply_files(ply_paths, num_points):
    """Merges three PLY files into one point cloud with specified colors and transformations."""
    merged_point_cloud = o3d.geometry.PointCloud()

    # Define colors (RGB values normalized between 0 and 1)
    # white_color = np.array([1.0, 1.0, 1.0])     # White
    white_color = np.array([1.0, 0.0, 0.0])     # White
    orange_color = np.array([1.0, 0.666, 0.0])  # #ffaa00

    for idx, ply_path in enumerate(ply_paths):
        if not os.path.exists(ply_path):
            print(f"Warning: File {ply_path} not found, skipping.")
            continue

        if idx == 2:  # Third file (end-effector mesh)
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(ply_path)
            # Apply orange color
            mesh.paint_uniform_color(orange_color)
            # Convert to point cloud
            mesh_point_cloud = mesh.sample_points_uniformly(number_of_points=5000)
            merged_point_cloud += mesh_point_cloud
        else:
            # Load the point cloud
            point_cloud = o3d.io.read_point_cloud(ply_path)
            if idx == 0:
                # Keep original RGB for the first file
                if not point_cloud.has_colors():
                    print(f"Warning: First file {ply_path} has no colors. Using default white.")
                    point_cloud.colors = o3d.utility.Vector3dVector(
                        np.tile(white_color, (np.asarray(point_cloud.points).shape[0], 1))
                    )
            elif idx == 1:
                # Color the second file white
                point_cloud.colors = o3d.utility.Vector3dVector(
                    np.tile(white_color, (np.asarray(point_cloud.points).shape[0], 1))
                )
            merged_point_cloud += point_cloud

    # Apply downsampling if num_points is specified
    if num_points > 0 and len(merged_point_cloud.points) > num_points:
        points = np.asarray(merged_point_cloud.points)
        indices = np.random.choice(len(points), num_points, replace=False)
        merged_point_cloud = merged_point_cloud.select_by_index(indices)

    # Apply a 180-degree rotation along the y-axis
    rotation_matrix = np.array([
        [-1, 0,  0, 0],  # Flip X
        [ 0, -1,  0, 0],  # Flip Y
        [ 0,  0, -1, 0],  # Flip Z
        [ 0,  0,  0, 1]
    ])
    merged_point_cloud.transform(rotation_matrix)

    # Apply a reflection along the XY plane
    reflection_matrix = np.array([
        [-1,  0,  0, 0],  # Keep X
        [ 0,  1,  0, 0],  # Keep Y
        [ 0,  0,  1, 0],  # Flip Z
        [ 0,  0,  0, 1]
    ])
    merged_point_cloud.transform(reflection_matrix)

    return merged_point_cloud

def load_ply_files(data_dir, num_points, left_hand):
    """Load and merge PLY files from scene, 3dhand, and transfer_hand_mesh directories."""
    scene_dir = os.path.join(data_dir, 'scene')
    hand_dir = os.path.join(data_dir, '3dhand')
    mesh_dir = os.path.join(data_dir, 'transfer_hand_mesh')

    if not all(os.path.exists(d) for d in [scene_dir, hand_dir, mesh_dir]):
        raise ValueError(f"One or more directories not found in {data_dir}: scene, 3dhand, transfer_hand_mesh")

    # Get list of scene files
    scene_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.ply')])
    if not scene_files:
        raise ValueError(f"No PLY files found in {scene_dir}")

    pcds = []
    ply_names = []
    hand_suffix = '_0' if left_hand else '_1'  # Left hand: _0, Right hand: _1
    for f in scene_files:
        name, ext = os.path.splitext(f)
        hand_f = f"{name}{hand_suffix}{ext}"
        ply_paths = [
            os.path.join(scene_dir, f),
            os.path.join(hand_dir, hand_f),
            os.path.join(mesh_dir, hand_f)
        ]
        try:
            merged_pcd = merge_and_color_ply_files(ply_paths, num_points)
            if len(merged_pcd.points) > 0:
                pcds.append(merged_pcd)
                ply_names.append(f)  # Use scene filename for display
            else:
                print(f"Warning: Merged point cloud for {f} is empty, skipping.")
        except Exception as e:
            print(f"Error processing {f}: {e}, skipping.")
            continue

    if not pcds:
        raise ValueError("No valid point clouds were loaded.")
    return pcds, ply_names

class PointCloudViewerApp:
    def __init__(self, pcds, ply_names, auto_mode, fps, skip_viz_frames):
        self.pcds = pcds
        self.ply_names = ply_names
        self.current_idx = 0
        self.fps = fps
        self.skip_viz_frames = max(1, skip_viz_frames)
        self.is_playing = auto_mode  # Auto mode if auto_mode is True
        
        # Get screen resolution
        width, height = get_screen_resolution()
        self.window = gui.Application.instance.create_window("Point Cloud Sequence Viewer", width, height)
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        
        # Option 2: Violet background (similar to MeshLab)
        self.scene.scene.set_background([0.3, 0.2, 0.4, 1.0])  # RGBA: Dark violet

        # Initialize scene
        self.update_geometry(self.current_idx)
        
        # Set up key event handling
        self.scene.set_on_key(self._on_key_event)
        
        # Configure rendering options
        self.scene.scene.set_lighting(self.scene.scene.LightingProfile.NO_SHADOWS, (0, 0, 1))
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        self.scene.scene.add_geometry("point_cloud", self.pcds[0], material)
        
        # Print initial frame info
        print(f"Frame {self.current_idx + 1}/{len(self.pcds)}: {self.ply_names[self.current_idx]}")
        
        # Set up animation for auto mode
        if self.is_playing:
            self.window.set_on_tick_event(self._on_tick)

    def update_geometry(self, idx):
        """Update the displayed point cloud."""
        self.scene.scene.clear_geometry()
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        self.scene.scene.add_geometry("point_cloud", self.pcds[idx], material)
        print(f"Frame {idx + 1}/{len(self.pcds)}: {self.ply_names[idx]}")
        self.scene.force_redraw()

    def _on_key_event(self, event):
        """Handle key events."""
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.RIGHT:
                if self.current_idx < len(self.pcds) - self.skip_viz_frames:
                    self.current_idx += self.skip_viz_frames
                    self.update_geometry(self.current_idx)
                self.is_playing = False
                return True
            elif event.key == gui.KeyName.LEFT:
                if self.current_idx >= self.skip_viz_frames:
                    self.current_idx -= self.skip_viz_frames
                    self.update_geometry(self.current_idx)
                else:
                    self.current_idx = 0
                    self.update_geometry(self.current_idx)
                self.is_playing = False
                return True
            elif event.key == gui.KeyName.SPACE:
                self.is_playing = not self.is_playing
                return True
        return False

    def _on_tick(self):
        """Handle animation tick for auto mode."""
        if self.is_playing:
            self.current_idx = (self.current_idx + self.skip_viz_frames) % len(self.pcds)
            self.update_geometry(self.current_idx)
        self.window.post_redraw()
        return 1000.0 / self.fps

def main():
    gui.Application.instance.initialize()
    args = get_args()
    pcds, ply_names = load_ply_files(Path(args.data_dir) / "out" / "hamer", args.num_points, args.left_hand)
    app = PointCloudViewerApp(pcds, ply_names, args.auto_mode, args.fps, args.skip_viz_frames)
    gui.Application.instance.run()

if __name__ == '__main__':
    main()