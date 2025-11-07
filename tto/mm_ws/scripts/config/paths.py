import os
import numpy as np

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
GRIPPER_OPEN_CLOSE_INFO_FILE = os.path.join(CURRENT_FILE_DIR, "gripper_open_close_info.json")

# Path to the demonstartion tasks root folder
DATA_DIR="/home/ash/Demonstrations"

# Path at which bundlesdf saves object pose transformation
OB_INCAM = "/home/robot-nav/Projects/mm-demo/vie/realworld/out"

# Is the task dual object based
IS_DUAL_OB = False
TASK_ID = 26

# Camera intrinsic parameters
if (1 < TASK_ID < 19) or (TASK_ID > 23):
    # tasks 2-18
    CAM_K = (
        527.8869068647631,
        524.7942507494529,
        321.7148665756361,
        230.2819198622499,
    )
else:
     # tasks 1, 19-22
    CAM_K = (
        574.0527954101562,
        574.0527954101562,
        319.5,
        239.5)

# Preprocessing step. Automatically move the robot within this distance to obj in straight line (optional)
DIST_ROBOT_OBJ = 1

IS_BASE = False # If True, base optimization is performed
IS_MASK = False # If True, query object mask to treat as a non obstacle region during optimization
IS_DELTA = False # do you want to apply object pose transformation or not
DO_PREPROCESSING = False
IS_SIM = True


"""
target_gripper_config:
    0: fetch gripper with old fingers + no ati
    1: fetch_gripper with old fingers + ati
    2: fetch_gripper with new umi fingers + ati
"""
CURRENT_GRIPPER_CONFIG = 0
TARGET_GRIPPER_CONFIG = 1

GRIPPER_PALM_WIDTH = 0.07
