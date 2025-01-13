#!/bin/bash

# Get the absolute path of the current directory
current_dir="/home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered"
current_dir="/home/jishnu/Projects/mm-demo/vie/_DATA/human_demonstrations-with-correct-depth"

conda activate robokit-py3.10

# Create an array of all directories
dirs=("$current_dir"/*/)

# Calculate the halfway point
halfway=$(( ${#dirs[@]} / 2 ))

# Iterate over the first half of the directories
for dir in "${dirs[@]:0:$halfway}"; do
    # Print the absolute path of the directory
    echo "Processing directory: $dir"

    CUDA_VISIBLE_DEVICES=0 python extract_hand_bboxes_and_meshes.py --input_dir $dir/rgb

    # cd ../rfp-grasp-transfer

    # Run the script and pass the directory as an argument
    # CUDA_VISIBLE_DEVICES=0 python transfer_from_hamer.py \
    # --mano_model_dir ../hamer/_DATA/data/mano/mano_v1_2/models/ \
    # --target_gripper fetch_gripper --debug_plots \
    # --input_dir $dir/rgb

    # cd ../hamer

done
