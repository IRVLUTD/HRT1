#!/usr/bin/env python

import os
import sys
import json
import rospy
import numpy as np
import toppra as ta
sys.path.insert(0, "..")
from copy import deepcopy
import toppra.algorithm as algo
import toppra.constraint as constraint
from config.transforms import RT_gripper_configs
from config.paths import GRIPPER_OPEN_CLOSE_INFO_FILE
from utils.utils import get_gaussian_interpolation_coefficients
from scipy.spatial.transform import Rotation as R

def extract_trajectory(
    data_dir_path,
    parent_frame="base_link",
    _3d_only=False,
    task_name="",
):
    """
    Method to return a trajectory as a list of 3d position lists, or list of RTmaxtrices
    Args:
        data_dir_path: path of the aligned demonstration data
                       Assuming data dir format as   Dir_name
                                                        |___depth
                                                        |___pose
                                                        |___rgb
                                                        |___out
                                                        |___task_description.txt
        parent_frame: Default "base_link". If "Camera_link" or any other, it will not be transformed into base_link
        _3d_only: Default value bool False. return list of lists [[x1,y1,z1],[x2,y2,z2],..]
                  If True, returns list of RTmatrices
    Returned trajectory is in parent_frame as a list of RT matrices
    """

    trajectory = []
    trajectory_poses_path = os.path.join(data_dir_path, "out/hamer/model")
    trajectory_poses_files = os.listdir(trajectory_poses_path)
    trajectory_poses_files.sort()

    poses_path = os.path.join(data_dir_path, "pose")

    with open(GRIPPER_OPEN_CLOSE_INFO_FILE, "r") as gripper_info_file:
        data = json.load(gripper_info_file)

    hand_flag = data[task_name]["hand_flag"]

    for pose_file in trajectory_poses_files:
        try:
            pose_data = np.load(os.path.join(trajectory_poses_path, pose_file))
            hand_flag_index = pose_data["right"].tolist().index(hand_flag)
            # cam frame
            pose = pose_data["target_transfer_pose"][hand_flag_index]
            if parent_frame == "base_link":
                pose_file = os.path.join(poses_path, pose_file)
                pose_data = np.load(pose_file)
                camera_pose = pose_data["RT_camera"]
                pose = np.dot(camera_pose, pose)
            if _3d_only:
                pose = pose[:3, 3]
            trajectory.append(pose)
        except:
            rospy.loginfo(f"Pose file {pose_file} data not considered.")

    return trajectory


def extract_trajectory_with_delta(
    data_dir_path,
    parent_frame="base_link",
    _3d_only=False,
    task_data_dir_path="",
    task_name="",
    RT_camera=np.eye(4, 4),
    obj_id=1
):
    """
    Method to return a trajectory as a list of 3d position lists, or list of RTmaxtrices
    Args:
        data_dir_path: path of the aligned demonstration data
                       Assuming data dir format as   Dir_name
                                                        |___depth
                                                        |___pose
                                                        |___rgb
                                                        |___out
                                                        |___task_description.txt
        parent_frame: Default "base_link". If "Camera_link" or any other, it will not be transformed into base_link
        _3d_only: Default value bool False. return list of lists [[x1,y1,z1],[x2,y2,z2],..]
                  If True, returns list of RTmatrices
    Returned trajectory is in parent_frame as a list of RT matrices
    """

    trajectory = []

    delta_pose_dir = os.path.join(data_dir_path, f"bundlesdf/rollout/obj_{obj_id}/ob_in_cam")
    delta_txtfiles = os.listdir(delta_pose_dir)
    delta_txtfiles.sort()

    RT_obj_pose1 = np.loadtxt(os.path.join(delta_pose_dir, delta_txtfiles[0]))
    RT_obj_pose2 = np.loadtxt(os.path.join(delta_pose_dir, delta_txtfiles[-1]))

    RT_delta = np.dot( RT_obj_pose2, np.linalg.inv(RT_obj_pose1))
    rospy.loginfo(f"Bundle SDF object delta pose {RT_delta}")

    trajectory_poses_path = os.path.join(task_data_dir_path, "out/hamer/model")
    trajectory_poses_files = os.listdir(trajectory_poses_path)
    trajectory_poses_files.sort()

    with open(GRIPPER_OPEN_CLOSE_INFO_FILE, "r") as gripper_info_file:
        data = json.load(gripper_info_file)

    hand_flag = data[task_name]["hand_flag"]

    for pose_file in trajectory_poses_files:
        try:
            pose_data = np.load(os.path.join(trajectory_poses_path, pose_file))
            hand_flag_index = pose_data["right"].tolist().index(hand_flag)
            # cam frame
            pose = pose_data["target_transfer_pose"][hand_flag_index]

            # ---------------------------------------------------------------- #
            # 1. Ref frame
            T_1o = RT_obj_pose1  # obj pose in ref cam frame
            T_1x = pose  # Traj (x) in ref cam frame
            T_ox = np.linalg.pinv(T_1o) @ T_1x  # traj in obj frame

            # 2. Target frame
            T_2o = RT_obj_pose2  # obj pose in target cam frame
            T_2x = T_2o @ T_ox  # traj in target cam frame
            # ---------------------------------------------------------------- #

            if parent_frame == "base_link":
                camera_pose = RT_camera  # realtime
                pose = np.dot(camera_pose, T_2x)
            if _3d_only:
                pose = pose[:3, 3]
            trajectory.append(pose)
        except Exception as e:
            rospy.loginfo(f"Pose file {pose_file} data not considered.")

    return trajectory

# extract object trajectory from the bundle sdf
def extract_object_trajectory(demo_path, obj_id=1):
    """
    Args:
        data_dir_path: path of the bundle sdf data of the object. listof text files
    Returns:
        object_trajectory: list of 4x4 RT matrices. each pose represents object pose in camera frame
    """
    object_pose_dir = os.path.join(demo_path, f"obj_{obj_id}/ob_in_cam")
    object_pose_files = os.listdir(object_pose_dir)
    object_pose_files.sort()
    object_trajectory = []

    for object_pose_file in object_pose_files:
        object_pose = np.loadtxt(os.path.join(object_pose_dir, object_pose_file))
        object_trajectory.append(object_pose)

    return object_trajectory

def get_object_delta(data_dir_path, obj_id=1):
    """
    Args:
        data_dir_path: path of the out directory
        obj_id: id of the object
    Returns:
        object_delta_pose: 4x4 RT matrix    
    """
    object_pose_dir = os.path.join(data_dir_path, f"bundlesdf/rollout/obj_{obj_id}/ob_in_cam")
    object_pose_files = os.listdir(object_pose_dir)
    object_pose_files.sort()
    RT_obj_pose1 = np.loadtxt(os.path.join(object_pose_dir, object_pose_files[0]))
    RT_obj_pose2 = np.loadtxt(os.path.join(object_pose_dir, object_pose_files[-1]))
    object_delta = np.dot( RT_obj_pose2, np.linalg.inv(RT_obj_pose1))
    return object_delta

def interpolate_trajectory(trajectory1, trajectory2, scale=4):
    mix, mix_rev = get_gaussian_interpolation_coefficients(len(trajectory1), scale=scale)
    trajectory_combined = []
    for i in range(len(trajectory1)):
        T1 = trajectory1[i]
        T2 = trajectory2[i]

        # --- Translation interpolation ---
        p1 = T1[:3, 3]
        p2 = T2[:3, 3]
        p = mix[i] * p1 + mix_rev[i] * p2

        # --- Rotation interpolation (SLERP) ---
        R1 = R.from_matrix(T1[:3, :3])
        R2 = R.from_matrix(T2[:3, :3])

        slerp = R.slerp(0, 1, [R1, R2])
        R_interp = slerp(mix[i]).as_matrix()

        # --- Combine into homogeneous matrix ---
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = p
        trajectory_combined.append(T_interp)
    # print(trajectory_combined)
    return trajectory_combined

def transform_trajectory(trajectory,
                         parent_frame_old_pose=np.eye(4, 4),
                         parent_frame_new_pose=np.eye(4, 4)):
    """
    The default trajectory is w.r.t parent_frame_old_pose.
    when parent frame is at a new pose ( i.e., parent frame had some movement),
        same trajectory in the world, is now represented w.r.t parent frame new pose
    Args:
        trajectory = list or array of 4x4 RT array
        parent_frame_old_pose = 4x4 RT array. Defaut = np.eye(4,4)
        parent_frame_new_pose = 4x4 RT array. Defaut = np.eye(4,4)
    returns: trnasformed trajectory w.r.t the choice of parent_frame_pose
    """
    transformed_trajectory = []
    RT_delta = np.linalg.inv(parent_frame_new_pose) @ parent_frame_old_pose
    for pose in trajectory:
        transformed_pose = RT_delta @ pose
        transformed_trajectory.append(transformed_pose)

    return transformed_trajectory

def filter_trajectory(
    trajectory,
    min_distance=0.01,
    max_distance=0.05,
):
    """
    Filter to eliminate outliers based on eucledian distance between successive points
    Idea is that: each point should be separated at least `min_distance` and atmost `max_distance` from it's neighbors
    Args:
        trajectory = list or array of 4x4 RT_matrices
        min_distance, max_distance; distance_thresholds (in meter).
    Returns the filtered trajectory
    """
    init_pose  = deepcopy(trajectory[0])
    # rm close and open gripper poses from trajectory to preserve
    _trajectory = trajectory[1:-1]
    filtered_trajectory = [init_pose]

    for i, pose in enumerate(_trajectory):
        # if i == 0:
        #     continue
        dist = np.linalg.norm(pose[:3,3] - filtered_trajectory[-1][:3,3])
        # print(f"distance of point {i} from last value {dist}")
        if min_distance <= dist <= max_distance:
            # print(f"last pose {filtered_trajectory[-1]} curent pose {pose}\n")
            # print(f"point {i} added")
            filtered_trajectory.append(pose)
            

    # Add the gripper open and close poses without change
    # filtered_trajectory.insert(0, ref_trajectory_og[0])
    filtered_trajectory.append(trajectory[-1])

    return filtered_trajectory


def translate_pose(pose, delta_translate=0.1, axis="x"):
    """
    translates the pose along its target axis
    Args:
        pose: numpy array 4x4
        delta_translate: distance to translate in meter
        axis: "x" or "y" or "z"
    returns translated pose 
    """
    axes = {"x": 0, "y": 1, "z": 2}
    target_axis = axes[axis]

    pose_axis_direction = pose[:3, target_axis]
    pose[:3, 3] += pose_axis_direction * delta_translate

    return pose


def align_trajectory_to_gripper_configuration(trajectory, current_gripper_config=0, target_gripper_config=0):
    """
    Aligns the existing trajectory which is defined when fetch had old fingers + no ati sensor
     to a new configuration
    Args:
        trajectory = list of 4x4 RT_arrays
        target_gripper_config:
                0: fetch gripper with old fingers + no ati
                1: fetch_gripper with old fingers + ati
                2: fetch_gripper with new umi fingers + ati
    Return aligned trajectory
    """
    # 1. Given trajectory is always assumed to be in w.r.t wrist roll link of current gripper config. transform into fingertip link
    trajectory_fingertip_current_config = [
            pose @ RT_gripper_configs[current_gripper_config] for pose in trajectory]
    # 2. Now we want the fingers of new config to be at same location.
    # So get the wristroll link pose of new configuration
    aligned_trajectory = []
    RT_inv = np.linalg.inv(RT_gripper_configs[target_gripper_config])
    for pose in trajectory_fingertip_current_config:
        pose = pose @ RT_inv
        aligned_trajectory.append(pose)
    return aligned_trajectory


def trim_trajectory(trajectory, start_from=20, end_at=-20):
    """
    Trim part of trajectory
    Args:
        trajectory = list of 4x4 RT matrices
        start_from = index from the start
        end_at = either +ve index from start or -ve index from the end
    Return:
        Trimmed trajectory
    """
    return trajectory[start_from:end_at]

def convert_plan_trajectory_toppra(plan, vlims, acclims):
    """
    Args:
        plan: numpy array of shape (ndof, T)
        vlims: numpy array of shape (ndof,)
        acclims: numpy array of shape (ndof,)
    Returns:
        qs_sample: numpy array of shape (ndof, num)
        qds_sample: numpy array of shape (ndof, num)
    """
    T = plan.shape[0]
    ss = np.linspace(0, 1, T)
    way_pts = plan
    
    path = ta.SplineInterpolator(ss, way_pts)
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(acclims)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()

    num = 100
    ts_sample = np.linspace(0, jnt_traj.duration, num)

    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)
    return qs_sample, qds_sample, qdds_sample, ts_sample