#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

import os
import cv2
import time
import rospy
import threading
import ros_numpy
import numpy as np
import message_filters
from PIL import Image as PILImg
from sensor_msgs.msg import Image as RosImage, CameraInfo
from robokit.ros.ros_utils import ros_qt_to_rt
import tf2_ros
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from utils import compute_xyz


class FetchImageListener:
    def __init__(self, camera="Fetch"):
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.trigger_flag = None
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mask_pub = rospy.Publisher("gsam/mask", RosImage, queue_size=10)
        self.image_pub = rospy.Publisher("gsam/overlay", RosImage, queue_size=10)

        if camera == "Fetch":
            self.base_frame = "base_link"
            rgb_sub = message_filters.Subscriber(
                "/head_camera/rgb/image_raw", RosImage, queue_size=10
            )
            depth_sub = message_filters.Subscriber(
                "/head_camera/depth_registered/image_raw", RosImage, queue_size=10
            )
            msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)
            self.camera_frame = "head_camera_rgb_optical_frame"
            self.target_frame = self.base_frame
        elif camera == "Realsense":
            self.base_frame = "measured/base_link"
            rgb_sub = message_filters.Subscriber(
                "/camera/color/image_raw", RosImage, queue_size=10
            )
            depth_sub = message_filters.Subscriber(
                "/camera/aligned_depth_to_color/image_raw", RosImage, queue_size=10
            )
            msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
            self.camera_frame = "measured/camera_color_optical_frame"
            self.target_frame = self.base_frame
        elif camera == "Azure":
            self.base_frame = "measured/base_link"
            rgb_sub = message_filters.Subscriber(
                "/k4a/rgb/image_raw", RosImage, queue_size=10
            )
            depth_sub = message_filters.Subscriber(
                "/k4a/depth_to_rgb/image_raw", RosImage, queue_size=10
            )
            msg = rospy.wait_for_message("/k4a/rgb/camera_info", CameraInfo)
            self.camera_frame = "rgb_camera_link"
            self.target_frame = self.base_frame
        else:
            self.base_frame = f"{camera}_rgb_optical_frame"
            rgb_sub = message_filters.Subscriber(
                f"/{camera}/rgb/image_color", RosImage, queue_size=10
            )
            depth_sub = message_filters.Subscriber(
                f"/{camera}/depth_registered/image", RosImage, queue_size=10
            )
            msg = rospy.wait_for_message(f"/{camera}/rgb/camera_info", CameraInfo)
            self.camera_frame = f"{camera}_rgb_optical_frame"
            self.target_frame = self.base_frame

        self.intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = self.intrinsics[0, 0]
        self.fy = self.intrinsics[1, 1]
        self.px = self.intrinsics[0, 2]
        self.py = self.intrinsics[1, 2]

        print("\n=================================================================\n")
        print(f"Camera intrinsics\n {self.intrinsics}")
        print("\n=================================================================\n")

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size, slop_seconds
        )
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                timeout=rospy.Duration(1.0),
            )
            trans = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
            rot = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
            RT_camera = ros_qt_to_rt(rot, trans)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            RT_camera = None

        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
            depth_cv = depth_cv * 1000
            depth_cv = depth_cv.astype(np.uint16)
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(1, f"Unsupported depth type: {depth.encoding}")
            return

        im = ros_numpy.numpify(rgb)
        with threading.Lock():
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera

    def get_latest_listener_data(self):
        with threading.Lock():
            if self.im is None:
                return None, None, None, None, None, None
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera.copy() if self.RT_camera is not None else None
            return (
                im_color,
                depth_img,
                rgb_frame_id,
                rgb_frame_stamp,
                RT_camera,
            )

    def get_gdino_preds(self, im_color, text_prompt):
        im = im_color.astype(np.uint8)
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, text_prompt)
        return bboxes, phrases, gdino_conf

    def get_gsam_mask(self, im_color, text_prompt):
        start_time = time.time()
        im = im_color.astype(np.uint8)
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, text_prompt)
        w, h = im.shape[1], im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, index = filter_large_boxes(
            image_pil_bboxes, w, h, threshold=0.5
        )
        masks = masks[index]
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]
        mask_time = time.time() - start_time
        return img_pil, masks, image_pil_bboxes, gdino_conf, phrases, mask_time

    def publish_overlay(self, im_label, rgb_frame_stamp, rgb_frame_id):
        rgb_msg = ros_numpy.msgify(RosImage, im_label, "rgb8")
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

    def publish_mask(self, mask, rgb_frame_stamp, rgb_frame_id):
        label_msg = ros_numpy.msgify(RosImage, mask.astype(np.uint8), "mono8")
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = "mono8"
        self.mask_pub.publish(label_msg)

    def save_mask(self, filename, img, depth, mask):
        rgb_path = os.path.join("/tmp", f"{filename}_rgb.jpg")
        depth_path = os.path.join("/tmp", f"{filename}_depth.png")
        mask_path = os.path.join("/tmp", f"{filename}_mask.png")
        cv2.imwrite(depth_path, depth)
        mask[mask > 0] = 255
        cv2.imwrite(mask_path, mask)
        PILImg.fromarray(img).save(rgb_path)
