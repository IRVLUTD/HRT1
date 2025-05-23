#!/bin/bash

# Base directory containing all captured data
# BASE_DIR="/home/jishnu/data_captured_rfpv1"
BASE_DIR=$1

# Find all 'rgb' directories and iterate over them
find "$BASE_DIR" -type d -name "rgb" | while read -r RGB_DIR; do
    echo "Processing: $RGB_DIR"
    python extract_hand_bboxes_and_meshes.py --input_dir "$RGB_DIR"
done

paplay /usr/share/sounds/freedesktop/stereo/bell.oga
