cd hamer
CUDA_VISIBLE_DEVICES=0 python extract_hand_bboxes_and_meshes.py --input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/scenereplica-10-122/rgb

cd ../rfp-grasp-transfer

# Run the script and pass the directory as an argument
CUDA_VISIBLE_DEVICES=0 python transfer_from_hamer.py \
--mano_model_dir ../hamer/_DATA/data/mano/mano_v1_2/models/ \
--target_gripper fetch_gripper --debug_plots \
--input_dir /home/jishnu/Projects/mm-demo/vie/_DATA/scenereplica-10-122/rgb
