export TOKENIZERS_PARALLELISM=True
PYTHONWARNINGS="ignore"
python run_object_pose_estimation.py --task_dir "$1" --text_prompt "$2" --src_frames $3 --realtime_frames $4
