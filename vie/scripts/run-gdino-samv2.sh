# CUDA_VISIBLE_DEVICES=1 python test_gdino_samv2.py --input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered/task_15_19s-use-knife/rgb  --text_prompt "knife"

# CUDA_VISIBLE_DEVICES=1 python test_gdino_samv2.py --input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered/task_16_12s-wipe-table-with-towel/rgb  --text_prompt "towel"

# CUDA_VISIBLE_DEVICES=1 python test_gdino_samv2.py --input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered/task_17_15s-squeeze-sponge-ball/rgb  --text_prompt "sponge ball"

# CUDA_VISIBLE_DEVICES=1 python test_gdino_samv2.py --input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/new-data-from-fetch-and-laptop/data_captured.filtered/task_18_10s-move-chair/rgb  --text_prompt "chair"

data_root_dir="/home/jishnu/Projects/mm-demo/vie/_DATA/human_demonstrations-with-correct-depth"

# remove _ suffixes from file names
# find "$data_root_dir" -type f -name "*_*.*" | while read -r file; do
#     dir=$(dirname "$file")                  # Get the directory of the file
#     base=$(basename "$file")                # Get the filename with extension
#     new_name="${base%%_*}.${base##*.}"      # Remove the suffix starting with `_`
#     mv "$file" "$dir/$new_name"             # Rename the file
# done


# rgb/ png -> jpg
find "$data_root_dir" -type f -path "*/rgb/*.png" | while read -r png_file; do
    jpg_file="${png_file%.png}.jpg"  # Replace .png with .jpg
    convert "$png_file" "$jpg_file" # Convert PNG to JPG
done

CUDA_VISIBLE_DEVICES=0 python test_gdino_samv2.py --input_dir $data_root_dir/whiteboard-eraser_interval_0.05/rgb  --text_prompt "black eraser"
CUDA_VISIBLE_DEVICES=0 python test_gdino_samv2.py --input_dir $data_root_dir/shelf-bottle_interval_0.05/rgb  --text_prompt "green bottle"
CUDA_VISIBLE_DEVICES=0 python test_gdino_samv2.py --input_dir $data_root_dir/microwave-open_interval_0.05/rgb  --text_prompt "white microwave door handle"
CUDA_VISIBLE_DEVICES=0 python test_gdino_samv2.py --input_dir $data_root_dir/fetch-shelf-ycb-red-mug_interval_0.05/rgb  --text_prompt "red mug"
CUDA_VISIBLE_DEVICES=0 python test_gdino_samv2.py --input_dir $data_root_dir/ecsn-water-fountain-pick-bottle-fill-water_interval_0.05/rgb  --text_prompt "green bottle"
