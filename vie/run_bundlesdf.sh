#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

export TOKENIZERS_PARALLELISM=True
PYTHONWARNINGS="ignore"
python run_bundlesdf.py --task_dir "$1" --demo_frames $2 --rollout_frames $3
