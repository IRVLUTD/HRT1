#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import os
import sys
import cv2
import time
import rospy
import argparse
import numpy as np
from fetch_listener import ImageListener
from std_msgs.msg import Bool


class DataSaver:
    def __init__(self, slop_seconds):
        rospy.init_node("data_saver_node", anonymous=True)
        self.slop_seconds = slop_seconds
        self.listener = ImageListener("Fetch", slop_seconds=self.slop_seconds)
        self.init_sleep = 2.5  # Initial sleep time to allow the listener to start properly
        self.time_delay = 0.1
        rospy.loginfo("Initializing... Sleeping for {} seconds".format(self.init_sleep))
        time.sleep(self.init_sleep)

        self.recording = False  # Flag to track recording state
        self.record_start_time = None  # To track the start time of the current recording
        self.data_count = 0  # Counter for saved files in the current session

        # Set the root directory for saving tasks
        self.root_dir = os.path.join(os.getcwd(), "data_captured")
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Initialize task counter based on existing directories
        self.task_count = self.get_latest_task_count()

        # Create the first task directory (will be recreated when start command is received)
        self.main_dir_name = None
        # self.create_directory_and_files()

        # Subscriber to the record command topic
        rospy.Subscriber("/hololens/out/record_command", Bool, self.record_command_callback)

    def get_latest_task_count(self):
        """Find the next available task counter based on existing directories."""
        if not os.path.exists(self.root_dir):
            return 0

        # Extract numeric parts of existing task directories
        existing_tasks = [
            int(dir_name.split("_")[1]) for dir_name in os.listdir(self.root_dir)
            if dir_name.startswith("task_") and dir_name.split("_")[1].isdigit()
        ]
        if existing_tasks:
            return max(existing_tasks) + 1
        return 0

    def create_directory_and_files(self):
        """Create a unique directory for the task under the root directory."""
        task_dir_name = "task_{}".format(self.task_count)
        self.main_dir_name = os.path.join(self.root_dir, task_dir_name)
        self.color_dir_name = os.path.join(self.main_dir_name, "rgb")
        self.depth_dir_name = os.path.join(self.main_dir_name, "depth")
        self.pose_dir_name = os.path.join(self.main_dir_name, "pose")

        os.makedirs(self.color_dir_name, exist_ok=True)
        os.makedirs(self.depth_dir_name, exist_ok=True)
        os.makedirs(self.pose_dir_name, exist_ok=True)

        np.savetxt(os.path.join(self.main_dir_name, "cam_K.txt"), self.listener.intrinsics, fmt="%.6f")
        rospy.loginfo("Created new task directory: {}".format(self.main_dir_name))

    def record_command_callback(self, msg):
        """Callback to handle recording commands."""
        if msg.data:  # Start recording
            if not self.recording:
                rospy.loginfo("Starting a new recording session.")

                self.recording = True
                self.record_start_time = rospy.get_time()  # Track start time
                self.data_count = 0  # Reset the data counter

                # Create a new task directory
                self.task_count = self.get_latest_task_count()
                self.create_directory_and_files()

            else:
                rospy.logwarn("Recording is already in progress. Ignoring start command.")
        else:  # Stop recording
            if self.recording:
                rospy.loginfo("Stopping the current recording session.")
                self.recording = False

                # Calculate duration and rename the directory
                duration = rospy.get_time() - self.record_start_time
                new_task_name = "{}_{}s".format(self.main_dir_name, int(duration))
                os.rename(self.main_dir_name, new_task_name)

                rospy.loginfo("Recording stopped. Duration: {:.2f} seconds.".format(duration))
                self.record_start_time = None  # Reset the start time

                # Prepare for the next recording session
                self.task_count += 1
            else:
                rospy.logwarn("No active recording to stop. Ignoring stop command.")

    def save_data(self):
        """Save data while recording."""
        while not rospy.is_shutdown():
            if self.recording:
                try:
                    rgb, depth, RT_camera = self.listener.get_data_to_save()

                    # Save pose data
                    np.savez(
                        os.path.join(self.pose_dir_name, "{:06d}.npz".format(self.data_count)),
                        RT_camera=RT_camera
                    )

                    # Save RGB and depth images
                    cv2.imwrite(os.path.join(self.color_dir_name, "{:06d}.jpg".format(self.data_count)), rgb) # jpg as SAMV2 needs jpg
                    cv2.imwrite(os.path.join(self.depth_dir_name, "{:06d}.png".format(self.data_count)), depth) # png else depth stored will be not accurate

                    rospy.loginfo("Saved data {}.".format(self.data_count))
                    self.data_count += 1
                except Exception as e:
                    rospy.logerr("Error while saving data: {}".format(e))
            rospy.sleep(self.time_delay)

    def __del__(self):
        rospy.loginfo("Shutting down SaveData node.")
        rospy.signal_shutdown("SaveData node terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch RGBD Listener with Hololens Stream")
    parser.add_argument("--slop_seconds", type=float, default=0.3, help="Slop for ApproximateTimeSynchronizer")
    args = parser.parse_args()
    data_saver = DataSaver(slop_seconds=args.slop_seconds)

    try:
        data_saver.save_data()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt detected. Exiting...")
        sys.exit(0)
