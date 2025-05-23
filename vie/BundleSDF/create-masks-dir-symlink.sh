#!/bin/bash

# Get the absolute path of the current directory
current_dir="/home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered"

# Activate the conda environment
conda activate robokit-py3.10

# Create an array of all subdirectories in the current directory
dirs=("$current_dir"/*/)

# Iterate over all the directories
for dir in "${dirs[@]}"; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Find the obj_masks directory under samv2
        obj_masks_path=$(find "$dir" -type d -path "*/samv2/*/obj_masks" -print -quit)

        # Check if the obj_masks directory was found
        if [ -n "$obj_masks_path" ]; then
            # Create a symlink named "masks" in the current root directory
            ln -sfn "$obj_masks_path" "${dir}masks"
            echo "Symlink created: ${dir}masks -> $obj_masks_path"
        else
            echo "No obj_masks directory found under $dir"
        fi
    fi
done
