# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# ✍️  Sai Haneesh Allu, Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import threading
import numpy as np
import rospy
import tf
import tf2_ros
import message_filters
import time
import yaml
import ros_numpy
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from nav_msgs.msg import Odometry, OccupancyGrid
from ros_utils import ros_qt_to_rt

lock = threading.Lock()


class ImageListener:
    def __init__(self, camera="Fetch", slop_seconds=0.3):
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        self.base_frame = "base_link"
        self.camera_frame = "head_camera_rgb_optical_frame"
        self.target_frame = self.base_frame
        self.slop_seconds = slop_seconds

        # Hololens publisher
        self.hololens_stream_pub = rospy.Publisher("/hololens_stream/compressed", CompressedImage, queue_size=1)

        # TF listener
        self.tf_listener = tf.TransformListener()

        # Subscribers
        rgb_sub = message_filters.Subscriber("/head_camera/rgb/image_raw", Image, queue_size=10)
        depth_sub = message_filters.Subscriber("/head_camera/depth_registered/image_raw", Image, queue_size=10)

        # Camera info
        msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx, self.fy = intrinsics[0, 0], intrinsics[1, 1]
        self.px, self.py = intrinsics[0, 2], intrinsics[1, 2]

        print("Camera intrinsics:", intrinsics)

        # Time synchronizer
        queue_size = 1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, self.slop_seconds)
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )
            RT_camera = ros_qt_to_rt(rot, trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None

        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
            depth_cv = (depth_cv * 1000).astype(np.uint16)
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).astype(np.float32) / 1000.0
        else:
            rospy.logerr_throttle(1, f"Unsupported depth type: {depth.encoding}")
            return

        im = ros_numpy.numpify(rgb)#[:, :, ::-1]  # bgr -> rgb

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, encoded_image = cv2.imencode('.jpg', im, encode_param)
            if not success:
                rospy.logwarn("JPEG encoding failed.")
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.rgb_frame_stamp
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_image.tobytes()
            self.hololens_stream_pub.publish(compressed_msg)
            rospy.loginfo("Published to /hololens_stream/compressed")
            rospy.sleep(0.1)

    def get_data_to_save(self):
        with lock:
            if self.im is None:
                return None, None, None
            return (
                self.im.copy(),
                self.depth.copy(),
                self.RT_camera.copy(),
            )

if __name__ == "__main__":
    rospy.init_node("fetch_image_listener")
    rospy.sleep(1) # Allow time for the node to initialize
    listener = ImageListener()
