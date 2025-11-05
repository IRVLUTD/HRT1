#!/usr/bin/env python
"""ROS image listener"""

import sys
sys.path.insert(0,"..")
import tf
import rospy
import tf2_ros
import threading
import ros_numpy
import numpy as np
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from utils.ros_utils import ros_qt_to_rt, odometry_to_rt

lock = threading.Lock()


class Listener:

    def __init__(self):

        self.rgb = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        # initialize a node before calling tflistener below
        self.tf_listener = tf.TransformListener()

        self.base_frame = "base_link"
        self.camera_frame = "head_camera_rgb_optical_frame"
        self.target_frame = self.base_frame
        self.base_frame_pose = np.eye(4,4)

        rgb_sub = message_filters.Subscriber(
            "/head_camera/rgb/image_raw", Image, queue_size=10
        )
        depth_sub = message_filters.Subscriber(
            "/head_camera/depth_registered/image_raw", Image, queue_size=10
        )
        odom_sub = message_filters.Subscriber(
            "/odom", Odometry, queue_size=10
        )
        msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)
        

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]

        queue_size = 5
        slop_seconds = 0.5
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, odom_sub], queue_size, slop_seconds
        )
        ts.registerCallback(self.callback_rgbd_joints_odom)


    

    def callback_rgbd_joints_odom(self, rgb, depth, odometry):

        # get camera pose in base
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )
            RT_camera = ros_qt_to_rt(rot, trans)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None

        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1,
                "Unsupported depth type. Expected 16UC1 or 32FC1, got {}".format(
                    depth.encoding
                ),
            )
            return

        rgb_cv = ros_numpy.numpify(rgb)
        with lock:
            self.rgb = rgb_cv.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera
            self.base_frame_pose = odometry_to_rt(odometry).copy()

    def compute_xyz(self, depth_img, fx, fy, px, py, height, width):
        indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
        return xyz_img

    
    def get_data(self, include_xyz = False):
        """
        Returns:
            rgb_image, depth_image, RT_camera, base_frame_pose  if include_xyz is False
            rgb_image, depth_image, RT_camera, base_frame_pose, xyz_image, xyz_base if include_xyz is True
        """
        with lock:
            if self.rgb is None:
                if include_xyz:
                    return None, None, None, None, None, None
                else:
                    return None, None, None, None
                
            rgb_image = self.rgb.copy()
            depth_image = self.depth.copy()
            RT_camera = self.RT_camera.copy()
            base_frame_pose = self.base_frame_pose.copy()

        if include_xyz:
            xyz_image = self.compute_xyz(
                depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width
            )
            xyz_array = xyz_image.reshape((-1, 3))
            xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(
                3, 1
            )
            xyz_base = xyz_base.T.reshape((self.height, self.width, 3))

            return rgb_image, depth_image, RT_camera, base_frame_pose, xyz_image, xyz_base
        else:
            return rgb_image, depth_image, RT_camera, base_frame_pose





if __name__ == "__main__":
    rospy.init_node("test_listener")
    listener = Listener()
    print(f"getting activated")
    rospy.sleep(3)
    for i in range(10):
        _, _, _,bpose= listener.get_data()
        rospy.sleep(0.2)
    print(bpose)
