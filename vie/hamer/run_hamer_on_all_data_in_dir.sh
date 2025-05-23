#!/bin/bash

# Root directory where all subdirectories are located
root_dir="/home/jishnu/Projects/mm-demo/vie/data/iteach-overlay-data-capture-5-8-25"

# Iterate over each subdirectory
for dir in "$root_dir"/*/; do
    # Check if the directory contains the 'rgb' folder
    if [ -d "$dir/rgb" ]; then
        echo "Running on directory: $dir/rgb"
        # Run the python script with the current directory's rgb folder as input
        python extract_hand_bboxes_and_meshes.py --input_dir "$dir/rgb"
    else
        echo "Skipping directory (no rgb folder found): $dir"
    fi
done
