#!/usr/bin/env python

import argparse
import os
import subprocess
import sys
from threading import Thread, Lock
import termios
import select
import tty
import time
import rospkg
import rospy
from datetime import datetime
from sensor_msgs.msg import JointState, Image, CameraInfo, Imu
from geometry_msgs.msg import Pose, WrenchStamped
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from moveit_msgs.msg import MoveItErrorCodes
import csv
import numpy as np
import message_filters
import ros_numpy
import cv2
import imageio
import tf
from tf.transformations import quaternion_matrix
from pathlib import Path
from PIL import Image as pli_Image


def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Data Recording Script for Teleop')
    parser.add_argument('--output_dir', type=str, help='Output directory of recordings',required=True)
    parser.add_argument('--f', type=int, help='Frequency to record with', default = 30)
    return parser

def getch():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class MoveItThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.start()

    def run(self):
        self.process = subprocess.Popen(["roslaunch", "fetch_moveit_config", "move_group.launch", "--wait"])
        _, _ = self.process.communicate()

    def stop(self):
        self.process.send_signal(subprocess.signal.SIGINT)
        self.process.wait()

def is_moveit_running():
    try:
        output = subprocess.check_output(["rosnode", "info", "move_group"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    if "unknown node" in output:
        return False
    if "Communication with node" in output:
        return False
    return True

def get_current_joint_values(joint_names):
    try:
        joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
        joint_values = {}
        for i, name in enumerate(joint_state.name):
            if name in joint_names:
                joint_values[name] = joint_state.position[i]
        return joint_values
    except rospy.ROSException:
        rospy.logerr("Could not retrieve joint states")
        return None

class FTRecorder(Thread):
    def __init__(self, frequency, output_dir,file_name = "ft_values.csv", topic_name = "/gripper/ft_sensor/absolute"):
        Thread.__init__(self)
        self.frequency = frequency  # Frequency in Hz (e.g., 10 Hz means 10 samples per second)
        self.output_file = os.path.join(output_dir, file_name)
        self.stop_flag = False
        self.lock = Lock()
        self.latest_ft_data = None
        
        # Initialize subscriber
        self.subscriber = rospy.Subscriber(topic_name, WrenchStamped, self.ft_state_callback)

        # Initialize CSV file
        with open(self.output_file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            # Write header: timestamp + joint names
            header = ['timestamp'] + ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
            writer.writerow(header)
        
        self.start()
    
    def ft_state_callback(self, msg):
        # Callback to update the latest joint values
        with self.lock:
            self.latest_ft_data = {
                'timestamp': msg.header.stamp.to_sec(),
                'fx': msg.wrench.force.x,
                'fy': msg.wrench.force.y,
                'fz': msg.wrench.force.z,
                'tx': msg.wrench.torque.x,
                'ty': msg.wrench.torque.y,
                'tz': msg.wrench.torque.z,
            }

    def run(self):
        rospy.loginfo("Starting FT value recording at" + str(self.frequency) + " Hz...")
        rate = rospy.Rate(self.frequency)  # ROS rate to control frequency

        while not rospy.is_shutdown() and not self.stop_flag:
            with self.lock:
                if self.latest_ft_data:
                    self.write_to_csv(self.latest_ft_data)
            rate.sleep()

        rospy.loginfo("FT recording stopped. Output saved to:" +  self.output_file)

    def write_to_csv(self, ft_values):
        with open(self.output_file, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            # Write row: timestamp + joint values
            row = [
                self.latest_ft_data['timestamp'],
                self.latest_ft_data['fx'],
                self.latest_ft_data['fy'],
                self.latest_ft_data['fz'],
                self.latest_ft_data['tx'],
                self.latest_ft_data['ty'],
                self.latest_ft_data['tz'],
            ]
            writer.writerow(row)

    def stop(self):
        self.stop_flag = True

class JointValueRecorder(Thread):
    def __init__(self, joint_names, frequency, output_dir):
        Thread.__init__(self)
        self.joint_names = joint_names
        self.frequency = frequency  # Frequency in Hz (e.g., 10 Hz means 10 samples per second)
        self.output_file = os.path.join(output_dir, "joint_state.csv")
        self.pose_output_file = os.path.join(output_dir, "finger_tip_pose.csv")  # CSV for pose
        self.stop_flag = False
        self.lock = Lock()
        self.latest_joint_values = None
        self.timestamp = None 
        self.tf_listener = tf.TransformListener()  # Initialize TF listener
        
        # Initialize subscriber
        self.subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        # Initialize joint state CSV file
        with open(self.output_file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            header = ['timestamp'] + self.joint_names
            writer.writerow(header)

        # Initialize pose CSV file with roll, pitch, yaw
        with open(self.pose_output_file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'timestamp', 
                'pos_x', 'pos_y', 'pos_z', 
                'quat_x', 'quat_y', 'quat_z', 'quat_w',
                'roll', 'pitch', 'yaw'
            ]
            writer.writerow(header)
        
        self.start()
    
    def joint_state_callback(self, msg):
        # Callback to update the latest joint values
        with self.lock:
            joint_values = {}
            for i, name in enumerate(msg.name):
                if name in self.joint_names:
                    joint_values[name] = msg.position[i]
            if "torso_lift_joint" in joint_values.keys():
                self.latest_joint_values = joint_values
                self.timestamp = msg.header.stamp.to_sec()

    def get_finger_tip_pose(self):
        # Compute the pose of finger_tip_link w.r.t. head_camera_rgb_frame
        try:
            # Wait for the transform to be available (timeout of 1 second)
            self.tf_listener.waitForTransform(
                "head_camera_rgb_optical_frame", "finger_tip_link", rospy.Time(0), rospy.Duration(1.0)
            )
            # Get the transform
            (trans, rot) = self.tf_listener.lookupTransform(
                "head_camera_rgb_optical_frame", "finger_tip_link", rospy.Time(0)
            )
            return trans, rot  # trans: [x, y, z], rot: [qx, qy, qz, qw]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed: %s" % str(e))
            return None, None

    def quaternion_to_euler(self, quat):
        # Convert quaternion [x, y, z, w] to roll, pitch, yaw (in radians)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quat)
        return roll, pitch, yaw

    def run(self):
        rospy.loginfo("Starting joint value and pose recording at " + str(self.frequency) + " Hz...")
        rate = rospy.Rate(self.frequency)  # ROS rate to control frequency

        while not rospy.is_shutdown() and not self.stop_flag:
            with self.lock:
                if self.latest_joint_values and self.timestamp:
                    # Write joint values
                    self.write_to_csv(self.timestamp, self.latest_joint_values)
                    # Compute and write finger tip pose with Euler angles
                    trans, rot = self.get_finger_tip_pose()
                    if trans is not None and rot is not None:
                        roll, pitch, yaw = self.quaternion_to_euler(rot)
                        self.write_pose_to_csv(self.timestamp, trans, rot, roll, pitch, yaw)
            rate.sleep()

        rospy.loginfo("Joint value and pose recording stopped. Output saved to: " + self.output_file + " and " + self.pose_output_file)

    def write_to_csv(self, timestamp, joint_values):
        with open(self.output_file, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            row = [timestamp] + [joint_values.get(name, None) for name in self.joint_names]
            writer.writerow(row)

    def write_pose_to_csv(self, timestamp, trans, rot, roll, pitch, yaw):
        with open(self.pose_output_file, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            # Write row: timestamp + position + quaternion + roll, pitch, yaw
            row = [timestamp] + list(trans) + list(rot) + [roll, pitch, yaw]
            writer.writerow(row)

    def stop(self):
        self.stop_flag = True

class IMURecorder(Thread):
    def __init__(self, frequency, output_dir):
        Thread.__init__(self)
        self.frequency = frequency  # Frequency in Hz (e.g., 10 Hz means 10 samples per second)
        self.output_file = os.path.join(output_dir, "imu_data.csv")
        self.stop_flag = False
        self.lock = Lock()
        self.latest_imu_data = None
        
        # Initialize subscriber 
        self.subscriber = rospy.Subscriber("/gripper/imu_raw", Imu, self.imu_state_callback)

        # Initialize CSV file
        with open(self.output_file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            # Write header: timestamp + format
            header = [
                'timestamp',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
            ]
            writer.writerow(header)
        
        self.start()
    
    def imu_state_callback(self, msg):
        # Callback to update the latest joint values
        with self.lock:
            self.latest_imu_data = {
                'timestamp': msg.header.stamp.to_sec(),
                'orientation': msg.orientation,
                'angular_velocity': msg.angular_velocity,
                'linear_acceleration': msg.linear_acceleration
            }

    def run(self):
        rospy.loginfo("Starting IMU data recording at" + str(self.frequency) + " Hz...")
        rate = rospy.Rate(self.frequency)  # ROS rate to control frequency

        while not rospy.is_shutdown() and not self.stop_flag:
            with self.lock:
                if self.latest_imu_data:
                    self.write_to_csv(self.latest_imu_data)
            rate.sleep()

        rospy.loginfo("IMU data recording stopped. Output saved to: "+ self.output_file)

    def write_to_csv(self, imu_data):
        with open(self.output_file, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            # Write row: timestamp + IMU data
            row = [
                imu_data['timestamp'],
                imu_data['orientation'].x,
                imu_data['orientation'].y,
                imu_data['orientation'].z,
                imu_data['orientation'].w,
                imu_data['angular_velocity'].x,
                imu_data['angular_velocity'].y,
                imu_data['angular_velocity'].z,
                imu_data['linear_acceleration'].x,
                imu_data['linear_acceleration'].y,
                imu_data['linear_acceleration'].z
            ]
            writer.writerow(row)

    def stop(self):
        self.stop_flag = True

class UnTuckThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.client = None
        self.start()
    def stop(self):
            # If stop pressing the button it should stop moving
            if self.client:
                self.client.get_move_action().cancel_goal()
            rospy.loginfo("Untuck Routine aborted")
            rospy.signal_shutdown("failed")
            sys.exit(1)
            return
    
    def run(self):
        move_thread = None
        if not is_moveit_running():
            rospy.logdebug("Starting MoveIt...")
            move_thread = MoveItThread()
        else:
            rospy.logdebug("MoveIt already started...")

        rospy.loginfo("Waiting for MoveIt...")
        self.client = MoveGroupInterface("arm_with_torso", "base_link")
        rospy.loginfo("...connected")
        
        # query the demo cube pose
        scene = PlanningSceneInterface("base_link")
        keepout_pose = Pose()
        keepout_pose.position.z = 0.375+0.11
        keepout_pose.orientation.w = 1.0
        ground_pose = Pose()
        ground_pose.position.z = -0.03
        ground_pose.orientation.w = 1.0
        rospack = rospkg.RosPack()
        mesh_dir = os.path.join(rospack.get_path('fetch_teleop'), 'mesh')
        scene.addMesh(
            'keepout', keepout_pose, os.path.join(mesh_dir, 'keepout.stl'))
        scene.addMesh(
            'ground', ground_pose, os.path.join(mesh_dir, 'ground.stl'))
        

        
        joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        
        # Get current joint values
        current_joint_values = get_current_joint_values(joints)
        while "torso_lift_joint" not in current_joint_values.keys():
            current_joint_values = get_current_joint_values(joints)
        
        if current_joint_values is None:
            rospy.logerr("Failed to get current joint values. Exiting.")
            sys.exit(1)

        pose = [current_joint_values["torso_lift_joint"], -0.845, -0.68, 0.413,
                0.882, 0.584, 1.41, 0.90]

        while not rospy.is_shutdown():
                result = self.client.moveToJointPosition(joints,
                                                        pose,
                                                        0.0,
                                                        max_velocity_scaling_factor=0.5)

                if result and result.error_code.val == MoveItErrorCodes.SUCCESS:
                    scene.removeCollisionObject("keepout")
                    scene.removeCollisionObject("ground")
                    if move_thread:
                        move_thread.stop()
                    
                    #rospy.signal_shutdown("done")
                    rospy.loginfo("Success in Reseting Position")
                    return
        
class RGBD_Recorder(Thread):

    def __init__(self, output_dir, camera='Fetch', frequency = 30):
        Thread.__init__(self)
        self.lock = Lock()
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.output_dir = os.path.join(output_dir, "video_data")
        self.frequency = frequency
        self.stop_flag = False

        # Lists to store RGB frames, depth frames, and timestamps
        self.rgb_frames = []
        self.depth_frames = []
        self.timestamps = []

        # Ensure output directory exists
        if (not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)


        if camera  == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Realsense':
            # use RealSense D435
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (camera)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (camera), Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (camera), Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (camera), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (camera)
            self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.rgb_video_writer = None

        # Time synchronizer for RGB and depth
        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        self.save_intrinsics(intrinsics)
        self.start()

    def save_intrinsics(self, intrinsics):
        # Save camera intrinsics to a text file
        intrinsics_file = os.path.join(self.output_dir, "camera_intrinsics.txt")
        with open(intrinsics_file, 'w') as f:
            f.write("Camera Intrinsics:\n")
            f.write(np.array2string(intrinsics))
        rospy.loginfo("Saved Intrinsics to " + intrinsics_file)

    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)

        with self.lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def run(self):
        rate = rospy.Rate(self.frequency)  # Frequency of recording
        rospy.loginfo("Starting RGBD recording at " + str(self.frequency) + " Hz...")
        while not rospy.is_shutdown() and not self.stop_flag:
            with self.lock:
                if self.im is not None and self.depth is not None:
                    self.save_frame(self.im, self.depth)
		    #self.save_frame_mp4_depricated(self.im, self.depth)
            rate.sleep()

        # Save frames
        #self.save_all_rgbd_frames()
	#self.rgb_video_writer.release()
     
    def save_frame(self, rgb_frame, depth_frame):
        # Save RGB frame as .jpeg
        rgb_file = os.path.join(self.output_dir, "rgb_"+str(self.counter)+".jpg")
        rgb_frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_file, rgb_frame_bgr)

        # Save depth frame as .png
        depth_file = os.path.join(self.output_dir, "depth_"+str(self.counter)+".png")
        depth_normalized = np.uint16(depth_frame * 1000)  # Convert to 16-bit integer (common for depth)
        cv2.imwrite(depth_file, depth_normalized)

        self.counter += 1

    def stop(self):
        self.stop_flag = True
        # files = sorted(Path(self.output_dir).glob('rgb_*'), key=lambda path: int(path.stem.split("_")[-1]))
        # imgs = [np.array(pli_Image.open(f)) for f in files]
        # frames = np.stack(imgs)
        # out = os.path.join(self.output_dir, "0_video.gif")
        # imageio.mimsave(out, frames, durations=frames.shape[0] / self.frequency)
        # rospy.loginfo("Gif video created and saved at " + out)
        
        rospy.loginfo("Stopping RGBD Recording")
        

        
        


if __name__ == "__main__":
    # intialize ros node
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('data_recorder_v0')
    parser = make_parser()
    args = parser.parse_args()
    out_dir = args.output_dir
    freq = args.f

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    # Recording Names
    joint_names = ["head_pan_joint", "head_tilt_joint", "torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint", "l_gripper_finger_joint",  "r_gripper_finger_joint"]

    keybindings = {
        'z': 'Start Recording',
        'x': 'Save Recording',
        # 'c': 'Press and hold for 3 seconds to place robot in origin position.',
        # 'v': 'Emergency Pause Robot Movement'
    }
    usage = 'Usage: '
    usage += ''.join('\n  {}: {}'.format(k, v)
                       for k, v in keybindings.items())
    usage += '\n  Ctrl-C to quit'

    try:
        rospy.loginfo(usage)
        untuck_thread = None
        joint_recorder = None
        rgbd_recorder = None
        ft_recorder = None
        imu_recorder = None
        ft_recorder_external = None
        rospy.loginfo("App Started")
        while True:
            c = getch()
            if c.lower() in keybindings:
                #print("Input:", c)
                if c == 'z':
                    
                    # JOint Value Record
                    if joint_recorder is None and rgbd_recorder is None and ft_recorder is None and imu_recorder is None and ft_recorder_external is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dir = os.path.join(out_dir,"Recordings_" + timestamp)
                        os.mkdir(dir)
                        joint_recorder = JointValueRecorder(joint_names, freq, dir)
                        rgbd_recorder = RGBD_Recorder(dir, frequency=freq)
                        ft_recorder = FTRecorder(output_dir=dir,frequency=freq)
                        ft_recorder_external = FTRecorder(output_dir=dir,frequency=freq, file_name = "ft_values_external.csv", topic_name = "/gripper/ft_sensor/external")
                        imu_recorder = IMURecorder(output_dir=dir, frequency=freq)
                    else:
                        rospy.loginfo("Recording in progress")
                elif c == 'x':
                    if joint_recorder is not None:
                        joint_recorder.stop()
                        joint_recorder.join()
                        joint_recorder = None
                    if ft_recorder is not None:
                        ft_recorder.stop()
                        ft_recorder.join()
                        ft_recorder = None
                    if ft_recorder_external is not None:
                        ft_recorder_external.stop()
                        ft_recorder_external.join()
                        ft_recorder_external = None
                    if imu_recorder is not None:
                        imu_recorder.stop()
                        imu_recorder.join()
                        imu_recorder = None
                    if rgbd_recorder is not None:
                        rgbd_recorder.stop()
                        rgbd_recorder.join()
                        rgbd_recorder = None
                    rospy.loginfo("Recording Complete")

                # elif c == 'c':
                #     start_time = time.time()

                #     while c == 'o':
                #         c = getch()  # Continuously check for keypress
                #         elapsed_time = time.time() - start_time
                #         if elapsed_time >= 3.0:
                #             if untuck_thread is None:
                #                 rospy.loginfo(" Starting UntuckThread...")
                #                 untuck_thread = UnTuckThread()
                #             held = True
                #         time.sleep(0.01)  # Small sleep to avoid busy-waiting

                # elif c == 'v':
                #     if untuck_thread is not None:
                #         rospy.loginfo(" Stopping UntuckThread...")
                #         untuck_thread.stop()
                #         # Destroy all recorders if they exist
                #         if joint_recorder is not None:
                #             joint_recorder.stop()
                #             joint_recorder.join()
                #             joint_recorder = None
                #         if ft_recorder is not None:
                #             ft_recorder.stop()
                #             ft_recorder.join()
                #             ft_recorder = None
                #         if ft_recorder_external is not None:
                #             ft_recorder_external.stop()
                #             ft_recorder_external.join()
                #             ft_recorder_external = None
                #         if imu_recorder is not None:
                #             imu_recorder.stop()
                #             imu_recorder.join()
                #             imu_recorder = None
                #         if rgbd_recorder is not None:
                #             rgbd_recorder.stop()
                #             rgbd_recorder.join()
                #             rgbd_recorder = None
                        
                #         raise(KeyboardInterrupt("Stopped Robot"))
                    # else:
                    #     rospy.loginfo(" No Untuck Routine Running...")

            else:
                if c == '\x03':
                    rospy.loginfo("Shutting Down ...")
                    if untuck_thread is not None:
                        untuck_thread.stop()
                    rospy.signal_shutdown("done")
                    if joint_recorder is not None:
                        joint_recorder.stop()
                        joint_recorder.join()
                        joint_recorder = None
                    if rgbd_recorder is not None:
                        rgbd_recorder.stop()
                        rgbd_recorder.join()
                        rgbd_recorder = None
                    if ft_recorder is not None:
                        ft_recorder.stop()
                        ft_recorder.join()
                        ft_recorder = None
                    if ft_recorder_external is not None:
                        ft_recorder_external.stop()
                        ft_recorder_external.join()
                        ft_recorder_external = None
                    if imu_recorder is not None:
                        imu_recorder.stop()
                        imu_recorder.join()
                        imu_recorder = None
                    break
                elif c == '\n'  or c == '\r':
                    rospy.loginfo("")
            
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
