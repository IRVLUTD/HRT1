#!/usr/bin/env python

import rospy
import ros_numpy
import threading
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import numpy as np

class ImageListener:
    def __init__(self):
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.height = None
        self.width = None
        self.lock = threading.Lock()

        # Since we are using only fetch camera, default pubs are for fetch camera
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.base_frame = 'base_link'

        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]

        queue_size = 5
        slop_seconds = 0.3
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):
        """Update image and depth from subscribers."""
        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
            depth_cv = depth_cv * 1000
            depth_cv = depth_cv.astype(np.uint16)
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float64)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1,
                "Unsupported depth type. Expected 16UC1 or 32FC1, got {}".format(depth.encoding),
            )
            return

        im = ros_numpy.numpify(rgb)  # BGR to RGB
        with self.lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]

    def get_image(self):
        """Get the current RGB image and metadata."""
        with self.lock:
            if self.im is None:
                return None, None, None
            return self.im.copy(), self.rgb_frame_id, self.rgb_frame_stamp
        
if __name__=="__main__":
    l = ImageListener()