#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
# This work is based on  https://irvlutd.github.io/GraspTrajOpt/ . 
# Credits to the authors of the original work for the code to start with.
#----------------------------------------------------------------------------------------------------
import sys
sys.path.insert(0, "..")
import os
import pdb
import json
import time
import datetime
import rospy
import argparse
import numpy as np
from copy import deepcopy
from scipy.io import loadmat
from planner_opt import TTOPlanner
from robotmodel_opt import TTORobotModel
from baseplanner_opt import BasePlanner
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
from utils.sim_utils import objectService
from utils.listener import Listener
from utils.control_utils import PointHeadClient, FollowTrajectoryClient
from utils.nav_utils import PIDController
from utils.viz_utils import publish_trajectory, publish_base_markers
from utils.moveit_utils import MotionPlanning, GripperPlanning, JointStateRecorder
from utils.utils import load_yaml, compute_object_pc_mean, transform_mean_position
from utils.ros_utils import rotZ, ros_qt_to_rt, rt_to_ros_qt, ros_pose_to_rt
from utils.traj_utils import (
    translate_pose,
    trim_trajectory,
    filter_trajectory,
    extract_trajectory,
    transform_trajectory,
    interpolate_trajectory,
    extract_trajectory_with_delta,
    convert_plan_trajectory_toppra,
    align_trajectory_to_gripper_configuration,
)
from utils.service_utils import MaskServiceClient
from config.paths import (
    DATA_DIR, TASK_ID,
    GRIPPER_OPEN_CLOSE_INFO_FILE,
    CAM_K, DIST_ROBOT_OBJ,
    IS_BASE, IS_DELTA, IS_MASK, IS_SIM, DO_PREPROCESSING,
    CURRENT_GRIPPER_CONFIG, TARGET_GRIPPER_CONFIG, GRIPPER_PALM_WIDTH, OB_INCAM
)



# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options to perform optimization')
    parser.add_argument('--debug_mode', type=bool, default=False, help='Enable debug mode (True/False)')
    parser.add_argument('--safe_mode', type=bool, default=False, help='Enable safe mode (True/False)')
    parser.add_argument('--stow_dir', type=str, choices=['left', 'right'], default='left', help='Stow direction (left/right)')
    parser.add_argument('--obj_prompt', type=str, default='cracker_box', help='Object text prompt')
    args = parser.parse_args()

    DEBUG_MODE = args.debug_mode
    SAFE_MODE = args.safe_mode
    STOW_DIR = args.stow_dir
    OBJ_PROMPT = args.obj_prompt

    rospy.loginfo("Starting optimization ROS node")
    rospy.init_node("track_traj")
    time.sleep(2)

    # 0. -------------- Initialize Planners and Publishers ---------------- ##

    listener = Listener()
    motionplanner = MotionPlanning()
    gripperplanner = GripperPlanning()
    jstate_recorder = JointStateRecorder(motionplanner=motionplanner)
    navplanner = PIDController(frequency=30)
    head_action = PointHeadClient()
    torso_action = FollowTrajectoryClient(
        "torso_controller", ["torso_lift_joint"])
    if IS_MASK:
        mask_request_client = MaskServiceClient()

    ## 0.1  Initialize Optas robot model
    robot_name = "fetch"
    base_effort_weight = 0.05
    root_dir = os.getcwd()
    config_file = os.path.join(
        root_dir, "data", "configs", f"{robot_name}.yaml")
        
    if not os.path.exists(config_file):
        print(f"robot {robot_name} not supported", config_file)
        sys.exit(1)
    cfg = load_yaml(config_file)["robot_cfg"]
    
    robot_model_dir = os.path.join(
        root_dir, "data", "robots", cfg["robot_name"])
    urdf_filename = os.path.join(robot_model_dir,"urdfs", cfg["robot_name"]+f"_config{TARGET_GRIPPER_CONFIG}.urdf")

    if DEBUG_MODE:
        rospy.loginfo(f"urdf filename {urdf_filename}")
        pdb.set_trace()

    robot = TTORobotModel(
        robot_model_dir,
        urdf_filename=urdf_filename,
        time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
        param_joints=cfg["param_joints"],
        collision_link_names=cfg["collision_link_names"],
    )
    
    ## -------------------------------------------------------------------- ## 

    rospy.loginfo(
        "listener, moveit and gripper planners, head controller, \n torso controller, mask client, optas robot model initialized")
    rospy.loginfo(
        f"IS_BASE {IS_BASE}\n IS_MASK {IS_MASK}\n IS_DELTA {IS_DELTA}\n IS_SIM {IS_SIM}\n DO_PREPROCESSING {DO_PREPROCESSING}\n")
    rospy.sleep(2) # Buffer time for planners, controllers to finish initialization

    # 1. ------------------- Preprocessing  ------------------------------- ##
    if IS_SIM:
        # Keep the fetch at origin for consistency in execution
        obj_service = objectService()
        obj_service.set_state("fetch", [0, 0, 0, 0, 0, 0, 1])
        QT_b4 = obj_service.get_state("fetch")
        RT_b4 = ros_qt_to_rt(QT_b4[3:], QT_b4[0:3])
        torso_action.move_to([0.4])
        head_action.look_at(0.45,0,0.75)

        mat_data = loadmat("scene_003_cracker_box.mat")
        spawn_position = mat_data["obj_pose"].flatten().tolist()
        obj_service.spawn_model("cafe_table",[1,0,-0.03,0,0,0,1],"") #0.75 for crackerbox
        rospy.sleep(1)
        obj_service.spawn_model("003_cracker_box", spawn_position,"")
        rospy.sleep(1)
        obj_service.set_state("003_cracker_box", spawn_position)
        rospy.loginfo(f"Spawned object: 003_cracker_box at pose: {spawn_position}")
        
    motionplanner.move_to_stow_position(direction=STOW_DIR)
    

    # 2. ------------------- Extract Trajectory --------------------------- ##
    ## 2.1 Read the task relevant folders
    task_prefix = f"task_{TASK_ID}"
    task_name = next((f for f in os.listdir(DATA_DIR) if f.startswith(task_prefix)), None)
    current_task_data_folder = os.path.join(DATA_DIR, task_name)

    ## 2.2 Read the gripper relevant data like when to open, close, where to trim the trajectory
    with open(GRIPPER_OPEN_CLOSE_INFO_FILE, "r") as gripper_info_file:
        data = json.load(gripper_info_file)
    gripper_checkpoints = [
        data[task_name]["gripper_close_frame"],
        data[task_name]["gripper_open_frame"],
    ]
    active_object_id = data[task_name]["active_object"]
    static_object_id = data[task_name]["static_object"]
    trajectory_ply_files_from_data = os.listdir(
        os.path.join(current_task_data_folder, "out/hamer/scene")
    )
    trajectory_indices_from_data = [
        int(item.split(".ply")[0]) for item in trajectory_ply_files_from_data
    ]
    trajectory_indices_from_data.sort()
    gripper_checkpoint_indices = [
        trajectory_indices_from_data.index(gripper_checkpoints[0]),
        trajectory_indices_from_data.index(gripper_checkpoints[1]),
    ]

    ## 2.3 Extract the trajectory with/without delta pose estimate
    """
    Here rather than transforming trajectory by taking current base position w.r.t odom,
    and then modifying the initial base position everywhere else in optimization, let's just keep the 
    trajectory assuming base is at np.eye(4,4), get the relative delta x,y and then go there first. 
    later transform the traj and provide the updated traj to tracking opt
    """
    im_color, depth_image, RT_camera, RT_base_b4_opt = listener.get_data()

    if IS_DELTA:
        ref_trajectory_active_obj = extract_trajectory_with_delta(
            data_dir_path=OB_INCAM, task_name=task_name, task_data_dir_path= current_task_data_folder, RT_camera=RT_camera,obj_id=active_object_id
        )
        ref_trajectory_active_obj = trim_trajectory(
            ref_trajectory_active_obj, start_from=gripper_checkpoint_indices[0], end_at=gripper_checkpoint_indices[1]-1
        )
        # 2.4 Get the trajectory of the object 2
        ref_trajectory_static_obj = extract_trajectory_with_delta(
            data_dir_path=OB_INCAM, task_name=task_name, task_data_dir_path= current_task_data_folder, RT_camera=RT_camera,obj_id=static_object_id
        )
        ref_trajectory_static_obj = trim_trajectory(
            ref_trajectory_static_obj, start_from=gripper_checkpoint_indices[0], end_at=gripper_checkpoint_indices[1]-1
        )
        # 2.5 Interpolate the trajectories
        ref_trajectory = interpolate_trajectory(ref_trajectory_active_obj, ref_trajectory_static_obj, scale=4)
    else:
        ref_trajectory = extract_trajectory(data_dir_path=current_task_data_folder, task_name=task_name)
        ref_trajectory = trim_trajectory(ref_trajectory, start_from=gripper_checkpoint_indices[0], end_at=gripper_checkpoint_indices[1]-1)

    filtered_trajectory = filter_trajectory(
        ref_trajectory,
        min_distance=0.5*GRIPPER_PALM_WIDTH,
        max_distance=2*GRIPPER_PALM_WIDTH,
    )

    # Align the filtered trajectory to desired gripper configuration
    aligned_trajectory = align_trajectory_to_gripper_configuration(
        filtered_trajectory, current_gripper_config=CURRENT_GRIPPER_CONFIG, target_gripper_config=TARGET_GRIPPER_CONFIG)

   # Add standoff pose
    standoff_pose = translate_pose(deepcopy(aligned_trajectory[0]), -0.2, axis="x") 
    aligned_trajectory.insert(0, standoff_pose)
    
    publish_trajectory(trajectory=aligned_trajectory,
                     config=TARGET_GRIPPER_CONFIG, duration=200)
    
    aligned_trajectory.insert(0, standoff_pose)
    aligned_trajectory.insert(0, standoff_pose)

    if DEBUG_MODE:
        rospy.loginfo(f"No of Trajectory points {len(aligned_trajectory)}")
        pdb.set_trace()
            
    # 3. -------------- Save pre optimization data ----------------------- ##

    data_save_folder_name = f'../experiment_data/{task_name}_{(datetime.datetime.now()).strftime("%c")}'
    os.makedirs(data_save_folder_name)
    np.savez(f"{data_save_folder_name}/primary_data.npz",
             ref_trajectory = aligned_trajectory,
             RT_base_b4_opt = RT_base_b4_opt,
             RT_camera = RT_camera
             )

    # 4. -----  Perform Base Optimization and move to goal position ------ ##

    if IS_BASE:
        if IS_MASK:
            mask = mask_request_client.call_mask_service("", OBJ_PROMPT )
            mask_request_client.save_and_display_mask(mask,"./mask.png")
    
            rospy.loginfo("Mask for Base optimization requested!")
            if mask is None:
                rospy.logerr("Got no mask! Try the script with different prompt/camera view! Exiting Now")
                sys.exit(1)
                
            object_mean_position, _, _ = compute_object_pc_mean(
                depth_image, mask, RT_camera, CAM_K, distance_threshold=DIST_ROBOT_OBJ)
        
        base_planner = BasePlanner(robot, cfg["link_ee"], cfg["link_gripper"])
        rospy.loginfo(f"Base planner for optimization initialized\n")
        depth_image[np.isnan(depth_image)] = 20  # np.inf  # TODO: check this
        depth_pc = DepthPointCloud(depth_image, listener.intrinsics, RT_camera)
        robot.setup_occupancy_grid(depth_pc.points)

        base_planner.setup_optimization(
            len(aligned_trajectory), base_effort_weight=base_effort_weight
        )

        robot_state = motionplanner.get_robot_state()
        joint_positions = robot_state.joint_state.position
        joint_names = robot_state.joint_state.name
        """
        joint names and robot.actuated_joint_names are of 15 length. But the ordering is different. since optimization planners use the robot model,
        we need to make sure we are sending the joint angles in the same order as queried from robot model.
        """
        q0 = np.array([joint_positions[joint_names.index(name)] for name in robot.actuated_joint_names]).reshape(-1, 1)
        plan, y, err_pos, err_rot, cost = base_planner.plan_goalset(
            q0, np.array(aligned_trajectory))
        rospy.loginfo(f"cost --> {cost}: If cost > 0 ==> no satisfactory solution found")

        # # show new base pose
        RT_base_delta = rotZ(y[2])
        RT_base_delta[0, 3] = y[0]
        RT_base_delta[1, 3] = y[1]
        RT_base_delta = np.linalg.inv(RT_base_delta)

        
        x_delta = RT_base_delta[0, 3] + 0.10 # small forward offset to compensate pid error due to its threshold
        y_delta = RT_base_delta[1, 3]
        theta_delta = -y[2]

        publish_base_markers([x_delta, y_delta, 0], duration=60)
        if DEBUG_MODE:
            rospy.loginfo(f"opt output  {[x_delta, y_delta, theta_delta]}")
            pdb.set_trace()
        if SAFE_MODE:
            input("Moving to optimized base location. Execute?")
        if not (x_delta < 0):
            navplanner.execute(x_delta, y_delta, theta_delta)
            rospy.loginfo("Sleeping for 3 seconds after reaching optimized base position...")
            rospy.sleep(3)

        im_color, depth_image, RT_camera, RT_base_a4tr_opt = listener.get_data()
        
        """
        In SIM base movement is not accurate due to in accurate friction modelling between wheels and surface.
        But you can see the actual movement in Rviz, which updates robot states based on the control signals
        Hence we set the state again to the optimized location. 
        But this is not an issue in real world
        """
        if IS_SIM:
            RT_delta = np.linalg.inv(RT_base_b4_opt) @ RT_base_a4tr_opt
            RT_a4 = RT_b4 @ RT_delta
            quat, trans = rt_to_ros_qt(RT_a4)
            QT_a4 = [*trans, *quat]
            obj_service.set_state("fetch", QT_a4)

        if IS_MASK:
            object_mean_position = transform_mean_position(
                    object_mean_position, RT_base_initial=RT_base_b4_opt, RT_base_final=RT_base_a4tr_opt)
            head_action.look_at(
                object_mean_position[0], object_mean_position[1], object_mean_position[2], "base_link")
            # This aligned trajectory is the same reference trajectory, but expressed from the new base position
        aligned_trajectory = transform_trajectory(
            aligned_trajectory, parent_frame_old_pose=RT_base_b4_opt, parent_frame_new_pose=RT_base_a4tr_opt)
        
        publish_trajectory(trajectory=aligned_trajectory,
                     config=TARGET_GRIPPER_CONFIG, duration=200)

        # 5. ----------- Save post base optimization data -------------------- ##

        np.savez(f"{data_save_folder_name}/secondary_data.npz",
                    opt_movement_delta=np.array([x_delta, y_delta, theta_delta]),
                    ref_trajectory_in_base_link_a4tr_opt=aligned_trajectory,
                    RT_base_b4_opt=RT_base_b4_opt,
                    RT_base_a4tr_opt=RT_base_a4tr_opt
                )

    # 6. --- After Grasp Optimization, perform Trajectory Optimization --- ##

    trajectory_opt_planner = TTOPlanner(
        robot,
        cfg["link_ee"],
        cfg["link_gripper"],
        T=len(aligned_trajectory),
        Tmax=100,
    )
    im_color, depth_image, RT_camera, RT_base_a4tr_opt = listener.get_data()
    depth_image[np.isnan(depth_image)] = 20
    depth_obstacle = depth_image.copy()

    if IS_MASK:
        mask = mask_request_client.call_mask_service("", OBJ_PROMPT )
        rospy.loginfo("Mask for trajectory optimization requested!")
        if mask is None:
                rospy.logerr("Got no mask! Try the script with different prompt/camera view! Exiting Now")
                sys.exit(1)

        object_indices = np.where(mask > 0)
        # we want object to be neglected as it is not obstacle. setting 20 gets treated as nan and not considered
        depth_obstacle[object_indices] = 20

    depth_pc = DepthPointCloud(depth_image, listener.intrinsics, RT_camera)
    robot.setup_points_field(depth_pc.points)
    world_points = robot.workspace_points
    sdf_cost_all = depth_pc.get_sdf_cost(world_points)

    depth_pc_obstacle = DepthPointCloud(
        depth_obstacle,
        listener.intrinsics,
        RT_camera,
        target_mask=None,
        threshold=cfg["depth_threshold"],
    )
    sdf_cost_obstacle = depth_pc_obstacle.get_sdf_cost(world_points)

    robot_state = motionplanner.get_robot_state()
    joint_positions = robot_state.joint_state.position
    joint_names = robot_state.joint_state.name
    """
    Reason to do below steps is that order of joints in robot model other than optimized joints is not the same as in joints queried from moveit robot state
    from moveit:
        ['l_wheel_joint', 'r_wheel_joint', 'torso_lift_joint', 'bellows_joint', 'head_pan_joint', 'head_tilt_joint', 
        'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 
        'l_gripper_finger_joint', 'r_gripper_finger_joint']
    from optas robot model:
        ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 
        'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 
        'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
    """
    q0 = np.array([joint_positions[joint_names.index(name)] for name in robot.actuated_joint_names]).reshape(-1, 1)
    qc = q0.flatten()

    start = time.time()
    plan, dQ, cost = trajectory_opt_planner.plan_ref_trajectory(
        qc, aligned_trajectory, sdf_cost_all, [0, 0, 0], interpolate=False
    )
    planning_time = time.time() - start
    print("plannnig time", planning_time, "cost", cost)

    # 7. ------------ Execute the Optimized Trajectory ------------------- ##

    """
    plan: 15xnum_waypoints
    plan_ros: 7xnum_waypoints

    """
    # plan_ros = plan[robot.optimized_joint_indexes, :] # optimized joints are the desired 7 joints from shoulder pan to wrist roll
    if DEBUG_MODE:
        rospy.loginfo(f"length of the plan {plan.shape[1]}")
        pdb.set_trace()
    if SAFE_MODE:
        input("Executing the optimized joint trajectory. Execute?")

    joint_indices = [joint_names.index(joint_name) for joint_name in robot.optimized_joint_names]
    tracked_wristroll_link_poses = []
    standoff_joint_angle = plan[:, 2][robot.optimized_joint_indexes]
    grasp_joint_angle = plan[:, 3][robot.optimized_joint_indexes]
    plan_post_grasp = plan[:, 3:].T

    # 7.1. move to stand off
    print(f"reaching standoff")
    motionplanner.set_velocity(0.5)
    motionplanner.move_to_joint_angle(standoff_joint_angle)
    # 7.2. move to grasp pose
    motionplanner.set_velocity(0.2)
    # is_success = motionplanner.compute_and_execute_cartesian_path(joint_angle=grasp_joint_angle, fraction_cutoff=0.9)
    # if not is_success:
    #     motionplanner.move_to_joint_angle(grasp_joint_angle)
    motionplanner.move_to_joint_angle(grasp_joint_angle)
    if SAFE_MODE:
        input("gripper closes here ! proceed?")
    gripperplanner.close()
    motionplanner.set_velocity(0.5)

    # 7.3. Add current state as start point of the post grasping plan
    current_state=motionplanner.get_robot_state()
    current_joint_position = np.array([[current_state.joint_state.position[i] for i in motionplanner.active_indices]])
    post_grasp_plan = np.vstack((current_joint_position, plan_post_grasp[:,robot.optimized_joint_indexes]))
    vlims = robot.velocity_optimized_joint_limits.toarray().flatten() * 0.2
    acclims = np.ones(post_grasp_plan.shape[1]) * 0.6
    q, qd, qdd, ts = convert_plan_trajectory_toppra(post_grasp_plan, vlims, acclims)
    trajectory = motionplanner.make_joint_trajectory(q, qd, qdd, ts)

    # 7.4. Record joint states and execute the trajectory
    jstate_recorder.start_recording()
    trajectory.header.stamp = rospy.Time.now() # this is a joint trajectory msg. 
    motionplanner.follow_trajectory(trajectory)
    jstate_recorder.stop_recording()
    gripperplanner.open()


    # 7.5. move to standoff post release
    wristroll_link_pose = motionplanner.get_joint_pose(joint_name="wrist_roll_link")
    wristroll_link_pose_array = ros_pose_to_rt(wristroll_link_pose.pose)
    exit_standoff_pose = translate_pose(wristroll_link_pose_array, -0.07, axis="x")
    is_success = motionplanner.compute_and_execute_cartesian_path(target_pose=exit_standoff_pose, fraction_cutoff=0.9)
    if not is_success:
        motionplanner.move_endeffector_to_pose(exit_standoff_pose)
    motionplanner.move_to_stow_position(direction=STOW_DIR)

    recorded_joint_states = jstate_recorder.joint_states
    for timestamp, joint_angle in recorded_joint_states:
        active_joints_angle = [joint_angle[i] for i in motionplanner.active_indices]
        wristroll_link_pose = motionplanner.get_fk_pose(active_joints_angle)
        tracked_wristroll_link_poses.append(ros_pose_to_rt(wristroll_link_pose.pose))
    
    # convert the data to save in list/array formart 
    recorded_joint_states_array = [list(joint_angle) for timestamp, joint_angle in recorded_joint_states]
    recorded_joint_states_timestamp_array = [timestamp.to_sec() for timestamp, joint_angle in recorded_joint_states]
    post_grasp_trajectory_array = [point.positions for point in trajectory.points]
    post_grasp_trajectory_timestamp_array = [point.time_from_start.to_sec() for point in trajectory.points]

    post_grasp_trajectory_start_time = trajectory.header.stamp.to_sec()
    rospy.loginfo(f"Saving Final Data !!!\n")
    np.savez(f"{data_save_folder_name}/final_data.npz",
                    ref_trajectory=aligned_trajectory,
                    optimized_plan = plan,
                    joint_indices=joint_indices,
                    optimized_post_grasp_plan=plan_post_grasp,
                    post_grasp_trajectory_start_time=post_grasp_trajectory_start_time,
                    post_grasp_trajectory_timestamp_array=post_grasp_trajectory_timestamp_array,
                    post_grasp_trajectory=post_grasp_trajectory_array,
                    recorded_joint_states_timestamp_array=recorded_joint_states_timestamp_array,
                    recorded_joint_states=recorded_joint_states_array,
                    tracked_wristroll_link_poses=tracked_wristroll_link_poses,
                )
    # 8. ------------ Stop the robot and shutdown the node -------------- ##
    motionplanner.stop()
    motionplanner.commander.roscpp_shutdown()
    if IS_SIM:
        obj_service.delete_model("cafe_table")
        obj_service.delete_model("003_cracker_box")

