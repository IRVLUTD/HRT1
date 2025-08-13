#!/bin/bash

# Get the absolute path of the current directory
data_root_dir=$1 #"/home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered"

conda activate robokit-py3.10

# Create an array of all directories
dirs=("$data_root_dir"/*/)

# Calculate the halfway point
halfway=$((${#dirs[@]} / 1))

# Iterate over the first half of the directories
for dir in "${dirs[@]:0:$halfway}"; do
    # Print the absolute path of the directory
    echo "Processing directory: $dir"

    python extract_hand_bboxes_and_meshes.py  --opt_weight 100.0 --input_dir $dir/rgb

    cd ../rfp-grasp-transfer

    # Run the script and pass the directory as an argument
    python transfer_from_hamer.py \
        --mano_model_dir ../hamer/_DATA/data/mano/mano_v1_2/models/ \
        --target_gripper fetch_gripper --debug_plots \
        --input_dir $dir

    cd ../hamer

done
