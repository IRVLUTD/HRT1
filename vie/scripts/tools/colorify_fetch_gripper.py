import open3d as o3d
import numpy as np
import os
from pathlib import Path

def color_fingers_by_y_axis(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ply_file in input_dir.glob("*.ply"):
        mesh = o3d.io.read_triangle_mesh(str(ply_file))
        mesh.compute_vertex_normals()
        
        vertices = np.asarray(mesh.vertices)
        colors = np.ones_like(vertices)

        # Simple heuristic: split by Y-axis to assign red/blue
        sorted_indices = np.argsort(vertices[:, 1])
        n = len(vertices)
        colors[sorted_indices[:n//2]] = [1, 0, 0]  # Red
        colors[sorted_indices[n//2:]] = [0, 0, 1]  # Blue

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        out_file = output_dir / ply_file.name
        o3d.io.write_triangle_mesh(str(out_file), mesh)
        print(f"Saved: {out_file}")

# Example usage
import sys
color_fingers_by_y_axis(sys.argv[1], sys.argv[2])

