#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    """
    Pseudocode:
    1. Check if a file path is provided as a command-line argument.
       - If not, print usage instructions and exit.
       
    2. Load the OBJ file from the specified path using the trimesh library.

    3. Compute a property of each vertex to use for coloring (e.g., height based on z-coordinate).
       - Extract the z-coordinates of the vertices to represent heights.

    4. Normalize the height values to a range of [0, 1].
       - This step is necessary to map the values to a colormap.

    5. Apply a colormap to the normalized height values.
       - Use a colormap from matplotlib (e.g., 'viridis') to get RGB colors for each vertex.
       - Extract only the RGB components and ignore the alpha channel.

    6. Assign the computed colors to the vertices of the mesh.
       - Convert the colors to the 0-255 range as required by trimesh's vertex color format.

    7. Display the mesh in an interactive viewer, where vertices are colored according to their height.
       - This creates a visual effect of spectrum colors mapped across the OBJ mesh.

    Usage:
        Run the script with a specified OBJ file path:
            python vis_obj.py path_to_your_file.obj color
    """

    # Check if the file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py path_to_your_file.obj color")
        sys.exit(1)

    # Get the file path from command-line arguments
    obj_file_path = sys.argv[1]

    # Load the OBJ file
    mesh = trimesh.load(obj_file_path)

    if len(sys.argv) == 3 and sys.argv[2] == 'color':
      # Compute a property to color by, such as vertex heights (z-coordinates)
      vertex_heights = mesh.vertices[:, 2]  # Z-coordinates for height

      # Normalize heights to range [0, 1] for colormap mapping
      normalized_heights = (vertex_heights - vertex_heights.min()) / (vertex_heights.max() - vertex_heights.min())

      # Apply a colormap (e.g., "plasma") to map heights to RGB colors
      colors = cm.plasma(normalized_heights)[:, :3]  # Get RGB values from colormap

      # Assign colors to vertices
      mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)  # Convert to 0-255 range for trimesh

    # Display the mesh with spectrum colors in an interactive viewer
    mesh.show()

if __name__ == "__main__":
    main()
