#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
#----------------------------------------------------------------------------------------------------
import sys
sys.path.insert(0,"..")

import math
import rospy
import numpy as np
from copy import deepcopy
from config.nav_pid import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from utils.ros_utils import ros_qt_to_rt, rotZ
from tf.transformations import euler_from_quaternion


class PIDController:
    def __init__(self, frequency = 10):
        """
        PID Controller for Navigation
        """
        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # In radians
        self.RT_odom_baselink = np.eye(4,4)
        self.RT_baselink_delta = np.eye(4,4)

        self.dt = 1/frequency
        self.rate = rospy.Rate(frequency)  # 10 Hz

        self.goal_x = 0
        self.goal_y = 0
        self.goal_theta = 0  # this is how robot should be aligned w.r.t current orientation (when at the goal position)
        self.linear_p_error = 0
        self.angular_p_error  = 0

        self.case_a = self.linear_p_error < LINEAR_TOLERANCE
        self.case_b = self.angular_p_error < ANGULAR_TOLERANCE

    def convert_local_to_global(self):
        """
        Converts a goal in local base_link frame to global odom frame
        """
        rot = deepcopy(self.orientation_q)
        rot = [rot.x, rot.y, rot.z, rot.w]
        trans = [deepcopy(self.current_x), deepcopy(self.current_y), 0]
        self.RT_odom_baselink = ros_qt_to_rt(rot, trans)
        goal_in_global_frame = np.dot(self.RT_odom_baselink, self.RT_baselink_delta)
        self.goal_x = goal_in_global_frame[0,3]
        self.goal_y = goal_in_global_frame[1,3]
        print("1.", f"goal x {self.goal_x}, goal y {self.goal_y}")
        
    def odom_callback(self, msg):
        # Get the robot's current position and orientation
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.orientation_q = msg.pose.pose.orientation
        _, _, self.current_theta = euler_from_quaternion(
            [self.orientation_q.x, self.orientation_q.y, self.orientation_q.z, self.orientation_q.w]
        )

    def execute(self, delta_x, delta_y, delta_theta):
        """
        Follows the sequence:
        1. Orient along line of sight to target
        2. Translate to target while also accounting for angular erros
        3. Finally, orient to target angle w.r.t initial position
        """

        self.RT_baselink_delta[0,3] = delta_x
        self.RT_baselink_delta[1,3] = delta_y
        # self.RT_baselink_delta[:3,:3] = rotZ(delta_theta)[:3,:3]
        self.convert_local_to_global()

        twist = Twist()       
        current_x, current_y = deepcopy(self.current_x), deepcopy(self.current_y)
        error_x = self.goal_x - current_x
        error_y = self.goal_y - current_y
        target_angular_error = math.atan2(delta_y, delta_x)
        # target_angular_error = math.atan2(error_y, error_x)

        self.goal_theta = deepcopy(self.current_theta) + delta_theta
        self.goal_theta = math.atan2(math.sin(self.goal_theta), math.cos(self.goal_theta))
        self.target_theta = deepcopy(self.current_theta) + deepcopy(target_angular_error)
        rospy.sleep(2)
        i=0
        while not rospy.is_shutdown():
            # Calculate errors
            current_x, current_y = deepcopy(self.current_x), deepcopy(self.current_y)
            error_x = self.goal_x - current_x
            error_y = self.goal_y - current_y
            # angular_delta_error = math.atan2(error_y, error_x) # how much robot should align to be in a straight line w.r.t goal position
            self.angular_p_error = self.target_theta - self.current_theta  # how much differnece between current anf final alignment when at goal positon
            self.linear_p_error = math.sqrt(error_x**2 + error_y**2)

            self.case_a = abs(self.linear_p_error) < LINEAR_TOLERANCE
            self.case_b = abs(self.angular_p_error) < ANGULAR_TOLERANCE
            twist.linear.x = LINEAR_Kp * self.linear_p_error * self.case_b * (i==0)
            twist.angular.z = ANGULAR_Kp * math.atan2(math.sin(self.angular_p_error), math.cos(self.angular_p_error)) 
            if self.case_a:
                twist.linear.x = 0
            if self.case_b:
                twist.angular.z = 0
            if (self.case_a and self.case_b) or (self.case_b and (i==1)):
                print(f"linear twist {twist.linear.x}")
                self.cmd_vel_pub.publish(twist)
                if i == 0:
                    self.target_theta = self.goal_theta
                    i+=1
                else:
                    break
            # Publish the velocities
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("test_pid")
        controller = PIDController()
        rospy.sleep(1)  # Wait for odom messages
        controller.execute(0.8,0,0)
        print(f"current theta {controller.current_theta}")
    except rospy.ROSInterruptException:
        rospy.loginfo("PID controller interrupted.")
