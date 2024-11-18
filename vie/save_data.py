#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu, Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
Run as python save_data.py <out-dirname> <fps (float is possible)>
Saves rgb, depth, camera poses
"""

import os
import sys
import cv2
import time
import rospy
import datetime
import numpy as np
from robokit.ros.fetch_listener import ImageListener

class SaveData:
    def __init__(self, task_name, time_interval):
        rospy.init_node("img_listen")
        self.listener = ImageListener("Fetch")
        self.init_sleep = 5
        rospy.loginfo("Sleep for %s seconds", self.init_sleep)
        time.sleep(self.init_sleep)
        self.task_name = task_name
        self.time_delay = time_interval
        self.create_directory()

    def create_directory(self):
        if self.task_name is None:
            # Create a directory named as the current date and time in the current working directory
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.main_dir_name = os.path.join(os.getcwd(), current_time)
        else:
            self.main_dir_name = os.path.join(os.getcwd(), self.task_name.lower().replace(' ','_'))
        self.color_dir_name = os.path.join(self.main_dir_name, "rgb")
        self.depth_dir_name = os.path.join(self.main_dir_name, "depth")
        self.pose_dir_name = os.path.join(self.main_dir_name, "pose")
        #self.map_dir_name = os.path.join(self.main_dir_name, "map")
        os.makedirs(self.main_dir_name)
        os.makedirs(self.color_dir_name)
        os.makedirs(self.depth_dir_name)
        os.makedirs(self.pose_dir_name)
        #os.makedirs(self.map_dir_name)

    def save_data(self):
        data_count = 0
        while not rospy.is_shutdown():
            rgb, depth, RT_camera, RT_laser, robot_velocity, RT_goal = self.listener.get_data_to_save()
            np.savez("{}.npz".format(os.path.join(self.pose_dir_name, "{:06d}".format(data_count))), RT_camera=RT_camera, robot_velocity=robot_velocity, RT_goal=RT_goal)
            
            # For map, uncomment
            # rgb, depth, RT_camera, RT_laser, RT_base, robot_velocity, RT_goal, map_data = self.listener.get_data_to_save()
            # np.savez("{}_pose.npz".format(os.path.join(self.pose_dir_name, "{:06d}".format(data_count))), RT_camera=RT_camera, RT_base=RT_base, robot_velocity=robot_velocity, RT_goal=RT_goal)
            
            cv2.imwrite("{}.jpg".format(os.path.join(self.color_dir_name, "{:06d}".format(data_count))), rgb)
            cv2.imwrite("{}.jpg".format(os.path.join(self.depth_dir_name, "{:06d}".format(data_count))), depth)
            #cv2.imwrite("{}_map.png".format(os.path.join(self.map_dir_name, "{:06d}".format(data_count))), map_data.astype(np.uint8))
            rospy.sleep(self.time_delay)
            data_count += 1

if __name__ == "__main__":
    saver = SaveData(str(sys.argv[1]), float(sys.argv[2]))
    saver.save_data()
