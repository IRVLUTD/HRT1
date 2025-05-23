import open3d as o3d
import numpy as np
import os
import argparse
from open3d.visualization import gui

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/ply_sequence/', help='Directory containing PLY files')
    parser.add_argument('--num_points', type=int, default=0, help='Number of points to sample from each point cloud (0 to use all points)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for auto mode')
    parser.add_argument('--skip_viz_frames', type=int, default=1, help='Show every Nth frame in auto mode or skip N frames per key press in manual mode (1 for all frames)')
    parser.add_argument('--mode', type=str, choices=['manual', 'auto'], default='manual', help='Navigation mode: manual (arrow keys) or auto (cyclic playback)')
    return parser.parse_args()

def load_ply_files(data_dir, num_points):
    """Load PLY files from the specified directory and return a list of point clouds."""
    ply_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.ply')])
    if not ply_files:
        raise ValueError(f"No PLY files found in {data_dir}")
    pcds = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(os.path.join(data_dir, ply_file))
        if num_points > 0 and len(pcd.points) > num_points:
            points = np.asarray(pcd.points)
            indices = np.random.choice(len(points), num_points, replace=False)
            pcd = pcd.select_by_index(indices)
        pcds.append(pcd)
    return pcds, ply_files

class PointCloudViewerApp:
    def __init__(self, pcds, ply_files, mode, fps, skip_viz_frames):
        self.pcds = pcds
        self.ply_files = ply_files
        self.current_idx = 0
        self.mode = mode
        self.fps = fps
        self.skip_viz_frames = max(1, skip_viz_frames)  # Ensure at least 1
        self.is_playing = (mode == 'auto')  # Start playing in auto mode
        self.window = gui.Application.instance.create_window("Point Cloud Sequence Viewer", 1920, 1080)  # Set to screen resolution
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        
        # Initialize scene with the first point cloud
        self.update_geometry(self.current_idx)
        
        # Set up key event handling
        self.scene.set_on_key(self._on_key_event)
        
        # Configure rendering options
        self.scene.scene.set_lighting(self.scene.scene.LightingProfile.NO_SHADOWS, (0, 0, 1))
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        self.scene.scene.add_geometry("point_cloud", self.pcds[0], material)
        
        # Print initial frame info
        print(f"Frame {self.current_idx + 1}/{len(self.pcds)}: {self.ply_files[self.current_idx]}")
        
        # Set up animation for auto mode
        if self.is_playing:
            self.window.set_on_tick_event(self._on_tick)

    def update_geometry(self, idx):
        """Update the displayed point cloud to the specified index."""
        self.scene.scene.clear_geometry()
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        self.scene.scene.add_geometry("point_cloud", self.pcds[idx], material)
        print(f"Frame {idx + 1}/{len(self.pcds)}: {self.ply_files[idx]}")
        self.scene.force_redraw()

    def _on_key_event(self, event):
        """Handle key events for navigation."""
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.RIGHT:
                if self.current_idx < len(self.pcds) - self.skip_viz_frames:
                    self.current_idx += self.skip_viz_frames
                    self.update_geometry(self.current_idx)
                self.is_playing = False  # Pause auto mode
                return True
            elif event.key == gui.KeyName.LEFT:
                if self.current_idx >= self.skip_viz_frames:
                    self.current_idx -= self.skip_viz_frames
                    self.update_geometry(self.current_idx)
                else:
                    self.current_idx = 0
                    self.update_geometry(self.current_idx)
                self.is_playing = False  # Pause auto mode
                return True
            elif event.key == gui.KeyName.SPACE:
                self.is_playing = not self.is_playing  # Toggle play/pause in auto mode
                return True
        return False

    def _on_tick(self):
        """Handle animation tick for auto mode."""
        if self.is_playing:
            self.current_idx = (self.current_idx + self.skip_viz_frames) % len(self.pcds)
            self.update_geometry(self.current_idx)
        self.window.post_redraw()
        return 1000.0 / self.fps  # Return interval in milliseconds

def main():
    # Initialize GUI application
    gui.Application.instance.initialize()
    
    # Parse command-line arguments
    args = get_args()
    
    # Load point clouds and file names
    pcds, ply_files = load_ply_files(args.data_dir, args.num_points)
    
    # Create and run the viewer application
    app = PointCloudViewerApp(pcds, ply_files, args.mode, args.fps, args.skip_viz_frames)
    gui.Application.instance.run()

if __name__ == '__main__':
    main()