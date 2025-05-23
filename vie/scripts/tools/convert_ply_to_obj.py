import open3d as o3d
from pathlib import Path
import sys

if len(sys.argv) != 2:
    print("Usage: python convert_ply_to_obj.py /path/to/ply_directory")
    sys.exit(1)

input_dir = Path(sys.argv[1])
if not input_dir.exists():
    print(f"Error: Directory {input_dir} does not exist.")
    sys.exit(1)

output_dir = input_dir.parent / (input_dir.name + "_obj")
output_dir.mkdir(parents=True, exist_ok=True)

for ply_file in input_dir.glob("*.ply"):
    mesh = o3d.io.read_triangle_mesh(str(ply_file))
    obj_file = output_dir / (ply_file.stem + ".obj")
    o3d.io.write_triangle_mesh(str(obj_file), mesh)
    print(f"Converted: {ply_file.name} -> {obj_file.name}")
