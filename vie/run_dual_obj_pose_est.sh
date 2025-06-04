export TOKENIZERS_PARALLELISM=True
PYTHONWARNINGS="ignore"
python run_dual_object_pose_estimation.py --task_dir "$1" --demo_frames $2 --rollout_frames $3
