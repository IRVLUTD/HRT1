#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Modified for HRT1 by Jishnu Jaykumar Padalunkal

import os
import cv2
import rospy
import shutil
import threading
import ros_numpy
import numpy as np
import message_filters
import argparse
from PIL import Image as PILImg
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.ros_utils import ros_qt_to_rt
import tf2_ros

lock = threading.Lock()

def draw_pose_axis(image, pose_matrix, intrinsic_matrix, axis_length=0.1, thickness=2, color_scheme=None):
    colors = color_scheme or [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    axis_points = np.array([
        [0, 0, 0, 1],
        [axis_length, 0, 0, 1],
        [0, axis_length, 0, 1],
        [0, 0, axis_length, 1]
    ]).T
    transformed_points = pose_matrix @ axis_points
    points_2d = intrinsic_matrix @ transformed_points[:3, :]
    points_2d /= points_2d[2]
    points_2d = points_2d[:2].T.astype(int)
    origin = tuple(points_2d[0])
    for i, color in enumerate(colors):
        image = cv2.line(image, origin, tuple(points_2d[i + 1]), color, thickness)
    return image

class DataCollectionNode:
    def __init__(self, task_dir, text_prompt, camera="Fetch", frames=15):
        self.task_dir = task_dir
        self.text_prompt = text_prompt
        self.camera = camera
        self.max_frames = frames
        self.current_frame = 1
        self.trigger_flag = None
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.RT_camera = None
        self.RT_goal = np.eye(4)
        self.robot_velocity = np.zeros(6)
        self.tf_listener = tf2_ros.Buffer()
        self.tf_listener_sub = tf2_ros.TransformListener(self.tf_listener)

        # Initialize network
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()

        # ROS initialization
        rospy.init_node("data_collection_node", anonymous=True)
        self.label_pub = rospy.Publisher("seg_label_refined", Image, queue_size=10)
        self.score_pub = rospy.Publisher("seg_score", Image, queue_size=10)
        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        self.ready_srv = rospy.Service("data_collection_ready", Trigger, self.handle_ready)

        # Subscribers
        rospy.Subscriber("/collect_mask_for_opt", String, self.trigger_for_save_pre_post_base_opt_mask_img)
        if camera == "Fetch":
            self.base_frame = "base_link"
            rgb_sub = message_filters.Subscriber("/head_camera/rgb/image_raw", Image, queue_size=10)
            depth_sub = message_filters.Subscriber("/head_camera/depth_registered/image_raw", Image, queue_size=10)
            msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)
            self.camera_frame = "head_camera_rgb_optical_frame"
            self.target_frame = self.base_frame
        elif camera == "Realsense":
            self.base_frame = "measured/base_link"
            rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=10)
            depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size=10)
            msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
            self.camera_frame = "measured/camera_color_optical_frame"
            self.target_frame = self.base_frame
        elif camera == "Azure":
            self.base_frame = "measured/base_link"
            rgb_sub = message_filters.Subscriber("/k4a/rgb/image_raw", Image, queue_size=10)
            depth_sub = message_filters.Subscriber("/k4a/depth_to_rgb/image_raw", Image, queue_size=10)
            msg = rospy.wait_for_message("/k4a/rgb/camera_info", CameraInfo)
            self.camera_frame = "rgb_camera_link"
            self.target_frame = self.base_frame
        else:
            self.base_frame = f"{camera}_rgb_optical_frame"
            rgb_sub = message_filters.Subscriber(f"/{camera}/rgb/image_color", Image, queue_size=10)
            depth_sub = message_filters.Subscriber(f"/{camera}/depth_registered/image", Image, queue_size=10)
            msg = rospy.wait_for_message(f"/{camera}/rgb/camera_info", CameraInfo)
            self.camera_frame = f"{camera}_rgb_optical_frame"
            self.target_frame = self.base_frame

        # Camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics

        # Synchronize RGB and depth
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.1)
        ts.registerCallback(self.callback_rgbd)

        # Setup directories
        self.setup_directories()

    def setup_directories(self):
        root_dir = os.path.dirname(os.path.normpath(self.task_dir))
        self.realworld_dir = "/realworld"
        rgb_source_dir = os.path.join(self.task_dir, "rgb")
        depth_source_dir = os.path.join(self.task_dir, "depth")
        pose_source_dir = os.path.join(self.task_dir, "pose")
        sam2_out_dir = os.path.join(self.task_dir, "out", "samv2")
        try:
            mask_source_dir = os.path.join(sam2_out_dir, os.listdir(sam2_out_dir)[-1], "obj_masks")
        except IndexError:
            rospy.logwarn(f"No directories found in {sam2_out_dir}. Skipping mask copy.")
            mask_source_dir = None
        self.realworld_rgb_dir = os.path.join(self.realworld_dir, "rgb")
        self.realworld_depth_dir = os.path.join(self.realworld_dir, "depth")
        self.realworld_pose_dir = os.path.join(self.realworld_dir, "pose")
        self.realworld_mask_dir = os.path.join(self.realworld_dir, "masks")
        cam_k_file_source = os.path.join(root_dir, "cam_K.txt")
        cam_k_file_target = os.path.join(self.realworld_dir, "cam_K.txt")

        os.makedirs(self.realworld_rgb_dir, exist_ok=True)
        os.makedirs(self.realworld_depth_dir, exist_ok=True)
        os.makedirs(self.realworld_pose_dir, exist_ok=True)
        os.makedirs(self.realworld_mask_dir, exist_ok=True)

        first_frame = "000000.jpg"
        first_frame_source = os.path.join(rgb_source_dir, first_frame)
        first_frame_target = os.path.join(self.realworld_rgb_dir, first_frame)
        if os.path.exists(first_frame_source):
            shutil.copy(first_frame_source, first_frame_target)
            rospy.loginfo(f"Copied {first_frame_source} to {first_frame_target}")
        else:
            rospy.logwarn(f"First frame {first_frame_source} does not exist.")

        first_frame_depth = "000000.png"
        first_frame_depth_source = os.path.join(depth_source_dir, first_frame_depth)
        first_frame_depth_target = os.path.join(self.realworld_depth_dir, first_frame_depth)
        if os.path.exists(first_frame_depth_source):
            shutil.copy(first_frame_depth_source, first_frame_depth_target)
            rospy.loginfo(f"Copied {first_frame_depth_source} to {first_frame_depth_target}")
        else:
            rospy.logwarn(f"Depth for first frame {first_frame_depth_source} does not exist.")

        if mask_source_dir and os.path.exists(mask_source_dir):
            first_frame_mask = "000000.png"
            first_frame_mask_source = os.path.join(mask_source_dir, first_frame_mask)
            first_frame_mask_target = os.path.join(self.realworld_mask_dir, first_frame_mask)
            if os.path.exists(first_frame_mask_source):
                shutil.copy(first_frame_mask_source, first_frame_mask_target)
                rospy.loginfo(f"Copied {first_frame_mask_source} to {first_frame_mask_target}")
            else:
                rospy.logwarn(f"Mask for first frame {first_frame_mask_source} does not exist.")
        else:
            rospy.logwarn("Mask source directory not available. Skipping mask copy.")

        first_frame_pose = "000000.npz"
        first_frame_pose_source = os.path.join(pose_source_dir, first_frame_pose)
        first_frame_pose_target = os.path.join(self.realworld_pose_dir, first_frame_pose)
        if os.path.exists(first_frame_pose_source):
            shutil.copy(first_frame_pose_source, first_frame_pose_target)
            rospy.loginfo(f"Copied {first_frame_pose_source} to {first_frame_pose_target}")
        else:
            rospy.logwarn(f"Pose for first frame {first_frame_pose_source} does not exist.")

        if os.path.exists(cam_k_file_source):
            shutil.copy(cam_k_file_source, cam_k_file_target)
            rospy.loginfo(f"Copied {cam_k_file_source} to {cam_k_file_target}")
        else:
            rospy.logwarn(f"{cam_k_file_source} does not exist.")

    def callback_rgbd(self, rgb, depth):
        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
            depth_cv = depth_cv * 1000
            depth_cv = depth_cv.astype(np.uint16)
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(1, f"Unsupported depth type. Expected 16UC1 or 32FC1, got {depth.encoding}")
            return

        im = ros_numpy.numpify(rgb)[:, :, ::-1]  # BGR to RGB
        try:
            trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
            RT_camera = ros_qt_to_rt(rot, trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            RT_camera = None

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera

    def run_network(self):
        with lock:
            if self.im is None:
                return None, None, None, None, None, None
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]  # RGB to BGR
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)
        w, h = im.shape[1], im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
        masks = masks[index]
        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]
        bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
        im_label = np.array(bbox_annotated_pil)

        label = mask
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), "mono8")
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = "mono8"
        self.label_pub.publish(label_msg)

        score = label.copy()
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        for index, mask_id in enumerate(mask_ids):
            score[label == mask_id] = gdino_conf[index]
        score_msg = ros_numpy.msgify(Image, score.astype(np.uint8), "mono8")
        score_msg.header.stamp = rgb_frame_stamp
        score_msg.header.frame_id = rgb_frame_id
        score_msg.encoding = "mono8"
        self.score_pub.publish(score_msg)

        rgb_msg = ros_numpy.msgify(Image, im_label, "rgb8")
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

        return im_color, depth_img, mask, self.RT_camera, self.RT_goal, self.robot_velocity

    def save_mask(self, filename, img, depth, mask):
        rgb_path = os.path.join("/tmp", f"{filename}_rgb.jpg")
        depth_path = os.path.join("/tmp", f"{filename}_depth.png")
        mask_path = os.path.join("/tmp", f"{filename}_mask.png")
        cv2.imwrite(depth_path, depth)
        mask[mask > 0] = 255
        cv2.imwrite(mask_path, mask)
        PILImg.fromarray(img).save(rgb_path)

    def trigger_for_save_pre_post_base_opt_mask_img(self, msg):
        self.trigger_flag = msg.data

    def handle_ready(self, req):
        if self.current_frame > self.max_frames:
            return TriggerResponse(success=True, message="Data collection complete")
        return TriggerResponse(success=False, message="Data collection in progress")

    def collect_real_time_data(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and self.current_frame <= self.max_frames:
            try:
                img, depth, mask, RT_camera, RT_goal, robot_velocity = self.run_network()
                if img is None:
                    continue

                rgb_path = os.path.join(self.realworld_rgb_dir, f"{self.current_frame:06d}.jpg")
                depth_path = os.path.join(self.realworld_depth_dir, f"{self.current_frame:06d}.png")
                mask_path = os.path.join(self.realworld_mask_dir, f"{self.current_frame:06d}.png")
                pose_path = os.path.join(self.realworld_pose_dir, f"{self.current_frame:06d}.npz")

                cv2.imwrite(depth_path, depth)
                mask[mask > 0] = 255
                cv2.imwrite(mask_path, mask)
                PILImg.fromarray(img).save(rgb_path)
                np.savez(pose_path, RT_camera=RT_camera, robot_velocity=robot_velocity, RT_goal=RT_goal)

                self.current_frame += 1
                rate.sleep()
            except Exception as e:
                rospy.logwarn(f"Error collecting frame {self.current_frame}: {e}")
                continue

    def pre_post_mask_capture(self):
        for tag in ["pre", "post"]:
            rospy.loginfo(f"Collecting mask for {tag} base optimization")
            while self.trigger_flag is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
            img, depth, mask, _, _, _ = self.run_network()
            if img is not None:
                self.save_mask(self.trigger_flag, img, depth, mask)
            self.trigger_flag = None

    def visualize_poses(self):
        final_file = f"{self.max_frames:06d}"
        img1 = cv2.imread(f"{self.realworld_rgb_dir}/000000.jpg")
        img2 = cv2.imread(f"{self.realworld_rgb_dir}/{final_file}.jpg")
        try:
            pose1 = np.loadtxt(f"{self.realworld_dir}/bundlesdf/ob_in_cam/000000.txt")
            pose2 = np.loadtxt(f"{self.realworld_dir}/bundlesdf/ob_in_cam/{final_file}.txt")
        except FileNotFoundError as e:
            rospy.logerr(f"Pose file missing: {e}")
            return
        K = np.loadtxt(f"{self.realworld_dir}/cam_K.txt")

        relative_pose_1 = np.linalg.inv(pose1) @ pose2
        img1_with_pose = draw_pose_axis(img1.copy(), pose1, K)
        img1_with_pose = draw_pose_axis(img1_with_pose, relative_pose_1, K, color_scheme=[(255, 0, 255), (255, 255, 0), (0, 255, 255)])
        img2_with_pose = draw_pose_axis(img2.copy(), pose1, K)
        img2_with_pose = draw_pose_axis(img2_with_pose, pose2, K)

        cv2.imwrite("000000_with_pose.png", img1_with_pose)
        cv2.imwrite(f"{final_file}_with_pose.png", img2_with_pose)
        rospy.loginfo(f"Poses drawn and saved as '000000_with_pose.png' and '{final_file}_with_pose.png'.")

    def run(self):
        self.collect_real_time_data()
        self.pre_post_mask_capture()
        self.visualize_poses()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Collection Node")
    parser.add_argument("--task_dir", type=str, required=True, help="Path to task directory")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for segmentation")
    args = parser.parse_args()

    try:
        camera = rospy.get_param("~camera", "Fetch")
        frames = rospy.get_param("~frames", 5)
        node = DataCollectionNode(args.task_dir, args.text_prompt, camera, frames)
        node.run()
    except rospy.ROSInterruptException:
        pass