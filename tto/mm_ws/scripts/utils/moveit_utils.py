#!/usr/bin/env python
#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
#----------------------------------------------------------------------------------------------------

import sys
sys.path.insert(0,"..")

import rospy
import actionlib
import threading
import numpy as np
import moveit_commander
from copy import deepcopy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import GripperCommandAction,GripperCommandGoal
from utils.ros_utils import xyz_euler_to_pose, rt_to_pose
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest

class MotionPlanning():
    """
    Planning module for the manipulator arm, consisting of various kinds of joint movements, obstacle add/del , velocty change etc.,
    Arg: group_name. default is "arm"
    """
    def __init__(self, group_name = "arm"):
        self.commander = moveit_commander
        self.commander.roscpp_initialize(sys.argv)
        self.robot = self.commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group_name)
        
        # forward kinematics service: joint angles -> desired joint pose w.r.t base link
        rospy.wait_for_service('/compute_fk', timeout=10.0)
        self.compute_fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        self.set_velocity(speed_scale=0.2)

        # Add base, netb as obstacles for moveit
        p = PoseStamped()
        p.header.frame_id = "base_link"
        
        p.pose.position.x = 0.22
        p.pose.position.y = 0
        p.pose.position.z = 0.4
        self.add_obstacle(name="netb", type="box", pose = p, size=[0.08, 0.2, 0.08])

        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0.18
        self.add_obstacle(name="base", type="box", pose = p, size=[0.58, 0.56, 0.4])
        
        # Add laptop as osbtacle on top of fetch head
        p.header.frame_id = "torso_lift_link"
        xyz = [-0.15,0,0.85]
        euler_angles = [0,0.6,0]
        pose = xyz_euler_to_pose(xyz, euler_angles)
        p.pose.position = pose.position
        p.pose.orientation = pose.orientation
        self.add_obstacle(name="laptop", type="box", pose = p, size=[0.2, 0.35, 0.3])

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                 DisplayTrajectory,
                                                 queue_size=20)

        # Initialize getting current state to avoid js13.2 
        for _ in range(3):
            self.robot.get_current_state()
        self.all_joint_names = self.robot.get_current_state().joint_state.name
        self.active_joint_names = self.group.get_active_joints()
        self.active_indices = [self.all_joint_names.index(joint) for joint in self.active_joint_names]

        self.fk_request = GetPositionFKRequest()
        self.fk_request.header.frame_id = self.group.get_planning_frame()
        self.fk_request.fk_link_names = [self.group.get_end_effector_link()]


    def add_obstacle(self, name="obstacle1", type="box", pose = PoseStamped(), size = [1,1,1], mesh_path=""):
        if type == "box":
            self.scene.add_box(name, pose, size)
        if type == "mesh":
            self.scene.add_mesh(name, pose, mesh_path)
    
    def remove_obstacle(self, name="obstacle1"):
        current_obstacles = self.scene.get_known_object_names()
        if name in current_obstacles:
            self.scene.remove_world_object(name)
        else:
            rospy.loginfo(f"obstacle {name} not present in the scene!")

    def clear_scene(self):
        self.scene.clear()
    
    def set_velocity(self, speed_scale):
        """
        Args: speed_scale -> fraction of max speed. between [0,1] 
        """
        self.group.set_max_velocity_scaling_factor(speed_scale)

    def stop(self):
        self.group.stop()
        self.group.clear_pose_targets()  

    def get_active_joint_angles(self):
        current_joint_angles = self.group.get_current_joint_values()
        return current_joint_angles
                
    def get_joint_pose(self, joint_name="wrist_roll_link"):
        """
        Get the current pose of the specified joint/link w.r.t base link
        Arg: Name of the joint or link to get the pose for
        return: geometry_msgs/PoseStamped msg representing the pose
        """
        try:
            pose = self.group.get_current_pose(joint_name)
            # rospy.loginfo(f"Pose of {joint_name}:\n {pose}")
            return pose
        except Exception as e:
            rospy.logerr(f"Failed to get pose for {joint_name}: {e}")
            return None
    
    def get_robot_state(self):
        """
        return: robot state having all joint pos, vel, accl
        """
        return self.robot.get_current_state()
    
    def get_fk_pose(self, joint_angle):
        """
        Arg: joint_angle of only active joints
        retun: PoseStamped msg of end effector psoe
        """
        robot_state = self.get_robot_state()
        robot_joint_positions = list(robot_state.joint_state.position)
        for i, index in enumerate(self.active_indices):
            robot_joint_positions[index] = joint_angle[i]
        robot_state.joint_state.position = robot_joint_positions 
        self.fk_request.robot_state = robot_state

        # Call FK service and get target pose
        fk_response = self.compute_fk(self.fk_request)
        return fk_response.pose_stamped[0]
    
    def filter_no_ik_grasps(self, grasps, get_immediate=False):
        """
        Arg: grasps: list of grasp poses array
        Returns: array of grasps with feasible ik
        """
        self.stop()
        filtered_grasps = []
        input_grasps = deepcopy(grasps)
        for i, grasp in enumerate(grasps):
            if isinstance(grasp, np.ndarray):
                grasp_pose = rt_to_pose(grasp)
            if isinstance(grasp, PoseStamped):
                grasp_pose = grasp.pose
            self.group.set_pose_target(grasp_pose)
            [feasibility, plan,_,_] = self.group.plan()
            print(f"feasibility {feasibility} for grasp {i}")
            if plan.joint_trajectory.points:
                if get_immediate:
                    return [grasp]
                filtered_grasps.append(input_grasps[i])
        return filtered_grasps
    
    def filter_no_ik_grasps_return_plan(self, grasps, get_immediate=False):
        """
        Arg: grasps: list of grasp poses array
        Returns: array of grasps with feasible ik
        """
        self.stop()
        filtered_grasps = []
        input_grasps = deepcopy(grasps)
        for i, grasp in enumerate(grasps):
            if isinstance(grasp, np.ndarray):
                grasp_pose = rt_to_pose(grasp)
            if isinstance(grasp, PoseStamped):
                grasp_pose = grasp.pose
            self.group.set_pose_target(grasp_pose)
            [feasibility, plan,_,_] = self.group.plan()
            print(f"feasibility {feasibility} for grasp {i}")
            if plan.joint_trajectory.points:
                if get_immediate:
                    return [grasp]
                filtered_grasps.append([plan.joint_trajectory.points[-1].positions ,input_grasps[i]])
        return filtered_grasps
    

    def move_to_joint_angle(self, target_joint_angle):
        """
        Arg: target_joint_angle : list of 7 joint angles (7dof)
        """
        self.stop()
        self.group.set_joint_value_target(target_joint_angle)
        self.group.go(wait=True)
        self.stop()

    def move_endeffector_to_pose(self, pose):
        """
        Arg: pose. list of [x, y, z, rotx, roty, rotz] or PoseStamped msg or 4x4 matrix
        This always moves the end effector link (wrist_roll_link) to specified pose
        """
        if isinstance(pose, np.ndarray):
            pose = rt_to_pose(pose)
        if isinstance(pose, PoseStamped):
            pose = pose.pose
        self.stop()
        self.group.set_pose_target(pose)
        [feasibility, plan,_,_] = self.group.plan()
        if plan.joint_trajectory.points:
            is_success = self.group.go(wait=True)
            rospy.loginfo(f"Has the end effector reached target pose : {is_success}")
        else:
            rospy.logwarn(f"No plan found for the target pose: {pose}")
            is_success = False
        self.stop()
        self.group.clear_pose_target("wrist_roll_link")

        return feasibility, is_success

    def move_to_stow_position(self, direction="right"):
        """
        Arg: direction "left" or "right"  when observed from behind the robot
        """
        p = PoseStamped()
        p.header.frame_id = "base_link"
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = -0.13
        # big box needed temporarily so arm doesn't hit the ground while stowing
        self.add_obstacle("big_base",type="box", pose=p, size=[1.3,1,1])
        if direction == "right":
            target_angle = [-1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.1]
        else:
            target_angle = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.1]
        self.move_to_joint_angle(target_angle)
        self.remove_obstacle("big_base")

    def display_trajectory(self, trajectory):
        """
        Arg: trajectory should be a moveit joint trajectory
        """
        robot_trajectory = RobotTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        robot_trajectory.joint_trajectory = trajectory
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(robot_trajectory)
        rospy.sleep(1)
        self.display_trajectory_publisher.publish(display_trajectory)

    def follow_trajectory(self, trajectory):
        """
        Arg: trajectory should be a moveit joint trajectory
        """
        self.trajectory = RobotTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        self.trajectory.joint_trajectory = trajectory
        self.stop()
        self.group.execute(self.trajectory, wait=True)
        self.stop()
    
    def make_display_joint_trajectory(self, trajectory_positions):
        """
        Arg: trajectory_positions. list of joint positions. each joint positions is a list of all 15 joints
        Method requires all 15 joints, to display accurately
        return: JointTrajectory msg of all joitns        
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.all_joint_names

        for i, position in enumerate(trajectory_positions):
            point = JointTrajectoryPoint()
            point.positions = position
            point.velocities = [0 for _ in range(len(self.all_joint_names))]
            point.accelerations = [0 for _ in range(len(self.all_joint_names))]
            point.time_from_start = rospy.Duration.from_sec(7*(i+1))
            trajectory.points.append(point)
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "base_link"
        return trajectory

    def make_joint_trajectory(self, trajectory_positions, trajectory_velocities, trajectory_accelerations, trajectory_durations, all_states=False):
        """
        Arg: trajectory_positions. list of joint positions. each joint positions is a list of all 15 joints
        Method selects only the active joints' postions to follow
        return: JointTrajectory msg of active joints        
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.active_joint_names
        if not all_states:
            active_joint_positions = trajectory_positions
            active_joint_velocities = trajectory_velocities
            active_joint_accelerations = trajectory_accelerations
            active_joint_durations = trajectory_durations
        else:
            active_joint_positions=[]
            for position in trajectory_positions:
                active_position = [position[index] for index in self.active_indices]
                active_joint_positions.append(active_position)
            active_joint_velocities = [0 for _ in range(len(self.active_joint_names))]
            active_joint_accelerations = [0 for _ in range(len(self.active_joint_names))]
            active_joint_durations = [rospy.Duration.from_sec(2*(i+1)) for _ in range(len(self.active_joint_names))]
        for i in range(len(active_joint_positions)):   
            point = JointTrajectoryPoint()
            point.positions = active_joint_positions[i]
            point.velocities = active_joint_velocities[i]
            point.accelerations = active_joint_accelerations[i]
            point.time_from_start = rospy.Duration.from_sec(active_joint_durations[i])
            trajectory.points.append(point)
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "base_link"
        return trajectory

    def compute_and_execute_cartesian_path(self,joint_angle=None, target_pose=None, fraction_cutoff=0.9):
        if isinstance(target_pose, np.ndarray):
            target_pose = rt_to_pose(target_pose)
        if isinstance(target_pose, PoseStamped):
            target_pose = target_pose.pose
        current_pose = self.get_joint_pose().pose
        if joint_angle is not None: 
            target_pose = self.get_fk_pose(joint_angle).pose
        waypoints = [current_pose, target_pose]
        (plan, fraction) = self.group.compute_cartesian_path(
                waypoints,           # List of poses
                eef_step=0.01,      
                avoid_collisions=False
            )
        if fraction < fraction_cutoff:
            rospy.logwarn(f"fraction for cartesian path: {fraction} is less than {fraction_cutoff}")
            return False
        rospy.loginfo(f"fraction for cartesian path: {fraction}")
        self.group.execute(plan, wait=True)
        self.stop()
        return True


class GripperPlanning():
    def __init__(self, group_name = "gripper"):
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.MIN_EFFORT = 35
        self.MAX_EFFORT = 100
        self.OPEN_POS = 0.8
        self.CLOSE_POS = 0

        self.action_client = actionlib.SimpleActionClient(
            "gripper_controller/gripper_action", GripperCommandAction 
        )
        self.action_client.wait_for_server(rospy.Duration(10))

    def open(self, position=None):
        goal = GripperCommandGoal()
        if position is None:
            goal.command.position = self.OPEN_POS
        else:
            goal.command.position = position
        self.action_client.send_goal_and_wait(goal, rospy.Duration(10))

    def close(self):
        goal = GripperCommandGoal()
        goal.command.position = self.CLOSE_POS
        goal.command.max_effort = self.MAX_EFFORT
        self.action_client.send_goal_and_wait(goal, rospy.Duration(10))


class JointStateRecorder:
    def __init__(self, motionplanner):
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.joint_states = []
        self.is_recording = False
        self.joint_state_lock = threading.Lock()
        self.last_recorded_time = None  # time of last recorded message
        self.target_interval = 0.1  #  ~10 Hz 
        self.motionplanner = motionplanner

    def joint_state_callback(self, msg):
        """
        rather than sleep for target interval, keep the message flowing in first
        when message timestamp in > time intrval, record it. better than blocking callback
        """
        if self.is_recording:
            current_time = rospy.get_time()
            with self.joint_state_lock:
                current_position = self.motionplanner.get_robot_state().joint_state.position
                if self.last_recorded_time is None or (current_time - self.last_recorded_time >= self.target_interval):
                    self.joint_states.append((rospy.Time.now(), deepcopy(current_position)))  # Store timestamp and copy of message
                    self.last_recorded_time = current_time
        else:
            rospy.sleep(0.01)

    def start_recording(self):
        """
        Start recording joint states.
        """
        with self.joint_state_lock:
            self.is_recording = True
            self.last_recorded_time = None  # Reset to record first message
        print("Recording started")

    def stop_recording(self):
        """
        Stop recording joint states.
        """
        with self.joint_state_lock:
            self.is_recording = False
            self.last_recorded_time = None
        print("Recording stopped")

if __name__=="__main__":
    from copy import deepcopy
    import numpy as np
    rospy.init_node("testmotionplanning")
    rospy.sleep(1)
    planner = MotionPlanning()
    rospy.sleep(3)
    recorder = JointStateRecorder()
    recorder.start_recording()
    rospy.sleep(1)
    recorder.stop_recording()
    rospy.sleep(1)
    print(recorder.joint_states[0])
    print(f"execute?")  
    for joint_angle in recorder.joint_states:
        wrist_link_pose = planner.get_fk_pose([joint_angle[1][i] for i in planner.active_indices])
        print(wrist_link_pose)
    print(planner.get_joint_pose(joint_name="wrist_roll_link").pose)
    ##--- 1. Add obstacle ---# 
    # p = PoseStamped()
    # p.header.frame_id = "base_link"
    # p.pose.position.x = -0.2
    # p.pose.position.y = 0
    # p.pose.position.z = 1.3
    # planner.add_obstacle(name="laptop", type="box", pose = p, size=[0.3955, 0.452, 0.3955])
    # rospy.sleep(2)
        
    # #--- 2. Gripper open/close ---#
    # gripper =GripperPlanning("gripper")
    # gripper.open()
    # gripper.close()
    # gripper.open()
    # rospy.sleep(2)

    # #--- 3. Stow/tuck the arm ---#
    # planner.move_to_stow_position("right")
    # rospy.sleep(2)

    # # --- 4. Move to a joint angle ---#
    # target_angles = [0,0,0,0,0,0,0]
    # current_angles = planner.get_active_joint_angles()
    # planner.move_to_joint_angle(target_joint_angle=target_angles)
    # planner.move_to_joint_angle(target_joint_angle=current_angles)
    # rospy.sleep(2)

    #--- 5. Move end effector to a pose ---# 
    # Remember: going to previous end effector position might not be same as going to previous joint angle
    # current_pose = planner.get_joint_pose(joint_name="wrist_roll_link")
    ## Case1: target_pose is a list of [x, y, z, rotx, roty, rotz]
        # target_pose = [1.9,0,0.9,0,0,0]
    ## Case2: target_pose is a 4x4 matrix
        # target_pose= np.eye(4)
        # target_pose[:3,:3] = np.array([[1.0000000,  0.0000000,  0.0000000],
        #                                 [0.0000000,  0.6795113, -0.7336650],
        #                                 [0.0000000,  0.7336650,  0.6795113]])
        # target_pose[:3,3] = np.array([0.9, 0, 0.9])
    ## Case3: target_pose is a PoseStamped msg
        # target_pose = PoseStamped()
        # target_pose.pose.position.x = 0.9
        # target_pose.pose.position.y = 0
        # target_pose.pose.position.z = 0.9
        # target_pose.pose.orientation.x = 0.4003
        # target_pose.pose.orientation.y = 0
        # target_pose.pose.orientation.z = 0
        # target_pose.pose.orientation.w = 0.9165
        # target_pose.header.frame_id = "base_link"
        # target_pose.header.stamp = rospy.Time.now()
    # print(planner.move_endeffector_to_pose(target_pose))
    # print(planner.move_endeffector_to_pose(current_pose))
    # rospy.sleep(2)
    
    # #--- 6. Follow a trajectory --#
    # # Note: This trajectory will always be w.r.t base_link. we need to change ref frame in rviz to base_link
    # trajectory_positions = []
    # current_state = planner.get_robot_state()
    # current_joint_position = list(current_state.joint_state.position)
    # target_joint_position = deepcopy(current_joint_position)
    # for active_index in planner.active_indices:
    #     target_joint_position[active_index] = 0
    # trajectory_positions.append(current_joint_position)
    # trajectory_positions.append(target_joint_position)
    # trajectory_positions.append(current_joint_position)

    # display_trajectory = planner.make_display_joint_trajectory(trajectory_positions)
    # arm_trajectory = planner.make_joint_trajectory(trajectory_positions)

    # planner.display_trajectory(display_trajectory)
    # rospy.sleep(3)
    # planner.follow_trajectory(arm_trajectory)
    # rospy.sleep(2)

    # #--- 7. Move in a straight line (cartesian path) ---#
    # current_angles = planner.get_active_joint_angles()
    # planner.compute_and_execute_cartesian_path(joint_angle=[0,0,0,0,0,0,0])
    # planner.move_to_joint_angle(target_joint_angle=current_angles)
    # rospy.sleep(2)

    # planner.clear_scene()


    ## --- 8. Get the wrist roll link pose ---#
    # from utils.ros_utils import ros_pose_to_rt
    # for _ in range(10):
    #     wristroll_link_pose = planner.get_joint_pose(joint_name="wrist_roll_link")
    #     print(wristroll_link_pose)
    #     wristroll_link_pose_array = ros_pose_to_rt(wristroll_link_pose.pose)
    #     np.savez(f"wristroll_link_pose_{_}.npz", wristroll_link_pose_array=wristroll_link_pose_array)
    #     rospy.sleep(2)

    # #--- 9. Filter no ik grasps ---#
    # # grasps = np.load("grasps.npy")
    # sample_grasps = [np.eye(4) for _ in range(4)]
    # sample_grasps[0] = np.array([[1.0, 0.0, 0.0, 0.9],
    #                              [0.0, 1.0, 0.0, 0.1],
    #                              [0.0, 0.0, 1.0, 0.8],
    #                              [0.0, 0.0, 0.0, 1.0]])
    # filtered_grasps = planner.filter_no_ik_grasps(sample_grasps)
    # print(f"filtered grasps: {filtered_grasps}")
    # rospy.sleep(2)

    # #--- 10. Backoff to standoff in a straightline  ---#
    # from utils.ros_utils import ros_pose_to_rt
    # from utils.traj_utils import translate_pose
    # wristroll_link_pose = planner.get_joint_pose(joint_name="wrist_roll_link")
    # wristroll_link_pose_array = ros_pose_to_rt(wristroll_link_pose.pose)
    # wristroll_link_pose_array = translate_pose(wristroll_link_pose_array, 0.3, axis="x")
    # planner.compute_and_execute_cartesian_path(target_pose=wristroll_link_pose_array, fraction_cutoff=0.9)
    # rospy.sleep(2)

    #--- 11. Move to a joint angle ---#
#     target_angles = [[1.3199362796776555, 0.7000644843429553, 0.00009223547976322942, -1.9999288060477456, 0.0000765052373843389, -0.5700352968418061, 0.10006701246771313],
#  [1.3199362796761054, 0.7000644843414053, 0.00009223547821348358, -1.9999288060492955, 0.00007650523583444861, -0.570035296843356, 0.10006701246616334],
#  [1.1728709140057572, -0.3656586356338191, 1.6887680830760226, -0.7494946855253132, 1.2768046059708806, -1.4192984541484632, 0.48786774662276217],
#  [0.8272282940918986, 0.0734322325972419, 0.17797072029730213, 0.0004745812397487401, -0.2821600601121899, 1.1699407798107901, -2.2155706635374033],
#  [0.8245851398396464, 0.1553449506641001, 0.23277170841164252, -0.17760313245140105, -0.2752675770268669, 1.2496297455545087, -2.295207658464889],
#  [0.9565552334343483, -0.12387433851884752, -0.9878601478380842, 0.7914167815181798, -1.8189693441166894, -1.0097614965257806, 0.13559850691539826],
#  [0.23204178392789565, 0.29517838614358266, -1.1585417975757142, -0.949136484214023, -2.013501982779729, -1.3593835312190756, 0.9859834326677521],
#  [0.1254935517298802, 0.32516407392910884, -1.047644778646514, -0.9007641603351872, -1.9395192081979935, -1.4383326786164559, 0.7885097955056672],
#  [-0.06859975039994133, 0.40046853420003853, -1.0628807743051414, -1.2001215828463205, -2.041247744670162, -1.5217736833762672, 0.8970886629002103],
#  [-0.21640368877328867, 0.40166416998838406, -1.0712963528626838, -1.1797288289412227, -1.9492036460249604, -1.4929162967466043, 0.7162653706253492],
#  [-0.2649100737105802, 0.2974588978356812, -1.1361369465190523, -0.9072450540333048, -1.8285233189697683, -1.429653098888083, 0.42382318553386844],
#  [-0.2552163946301831, 0.23374227566784478, -1.1190631273859697, -0.6258782786033062, -1.7795052875676403, -1.3966938455371993, 0.12684964664796855],
#  [-0.15572281718880196, 0.12863447416778262, 1.9494437306200114, 0.2586603371002661, 1.3466708706295014, -1.2272066908079386, -0.2922802905733011],
#  [-0.07980270394839735, 0.08762983042243636, 2.0007514645021263, 0.00010237621044744357, 1.398131385818431, -1.175489131777771, -0.5404203310032826]]
#     for i, angle in enumerate(target_angles):
#         planner.move_to_joint_angle(angle)
#         rospy.loginfo(f"reached angle {i}")
#     planner.move_to_stow_position()

    # a=[0,0,0,0,0,0,0]
    # pose = planner.get_fk_pose(a)
    # print(pose)
    # print(type(pose))


