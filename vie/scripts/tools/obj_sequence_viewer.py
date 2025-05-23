import open3d as o3d
import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/ply_sequence/')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--vis_downsample', type=int, default=1)
    return parser.parse_args()

def load_ply_files(data_dir, num_points):
    ply_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.ply')])
    pcds = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(os.path.join(data_dir, ply_file))
        if num_points > 0:
            points = np.asarray(pcd.points)
            indices = np.random.choice(len(points), num_points, replace=False)
            pcd = pcd.select_by_index(indices)
        pcds.append(pcd)
    return pcds

def main():
    args = get_args()
    pcds = load_ply_files(args.data_dir, args.num_points)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcds[0])
    
    def update(vis, pcds, idx, wait_time):
        idx = (idx + 1) % len(pcds)
        vis.clear_geometries()
        vis.add_geometry(pcds[idx])
        vis.poll_events()
        vis.update_renderer()
        vis.register_animation_callback(lambda vis: update(vis, pcds, idx, wait_time))
    
    wait_time = 1000.0 / args.fps
    vis.register_animation_callback(lambda vis: update(vis, pcds, 0, wait_time))
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()