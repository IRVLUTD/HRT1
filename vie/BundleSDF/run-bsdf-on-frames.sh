#!/bin/bash

# Get the absolute path of the current directory
current_dir="/home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered"

conda activate robokit-py3.10

# Create an array of all directories
dirs=("$current_dir"/*/)

# Calculate the halfway point
# halfway=$(( ${#dirs[@]} / 2 ))

# Iterate over the first half of the directories
for dir in "${dirs[@]}"; do
    # Print the absolute path of the directory
    echo "Processing directory: $dir"

    CUDA_VISIBLE_DEVICES=1 python run_custom.py --mode run_video --video_dir $dir --out_folder $dir/out/bundlesdf --use_segmenter 0 --use_gui 1 --debug_level 2

done
