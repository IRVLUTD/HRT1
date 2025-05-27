#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Merged script combining ros_test_images.py and BundleSDF/run_custom.py
# Runs ros_test_images.py first to collect data, then processes with BundleSDF

import os
import cv2
import sys
import time
import rospy
import shutil
import threading
import ros_numpy
import numpy as np
import message_filters
import argparse
import glob
import imageio
import logging
from PIL import Image as PILImg
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage, CameraInfo
from robokit.ros.fetch_listener import ImageListener
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.ros.ros_utils import ros_qt_to_rt
import tf2_ros
from geometry_msgs.msg import TransformStamped
import ros_numpy as rnp

# Explicitly import PyYAML to avoid conflicts
try:
    import yaml as pyyaml
except ImportError:
    print("Error: PyYAML not installed. Install it with 'pip install pyyaml'")
    sys.exit(1)

# Import gdown for Google Drive downloads
try:
    import gdown
except ImportError:
    print("Error: gdown not installed. Installing it now...")
    os.system("pip install gdown")
    import gdown

# Adjust sys.path to include BundleSDF subdirectory
code_dir = os.path.dirname(os.path.realpath(__file__))
bundle_sdf_dir = os.path.join(code_dir, "BundleSDF")
sys.path.append(bundle_sdf_dir)

try:
    from bundlesdf_original import *
    from segmentation_utils import Segmenter
except ModuleNotFoundError as e:
    print(f"Error importing BundleSDF modules: {e}")
    print(
        "Ensure 'my_cpp' module is built in BundleSDF directory. Run 'cmake .. && make' in BundleSDF/build/"
    )
    sys.exit(1)

lock = threading.Lock()


def set_seed(seed):
    np.random.seed(seed)


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
    return xyz_img


def download_loftr_weights(
    weights_dir, drive_folder_id="1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp"
):
    """Download outdoor_ds.ckpt from Google Drive if it doesn't exist."""
    weights_path = os.path.join(weights_dir, "outdoor_ds.ckpt")
    if os.path.exists(weights_path):
        print(f"LoFTR weights already exist at {weights_path}")
        return
    print(
        f"LoFTR weights not found at {weights_path}. Downloading from Google Drive..."
    )
    os.makedirs(weights_dir, exist_ok=True)
    folder_url = f"https://drive.google.com/drive/folders/{drive_folder_id}"
    try:
        gdown.download_folder(folder_url, output=weights_dir, quiet=False)
        if os.path.exists(weights_path):
            print(f"Successfully downloaded outdoor_ds.ckpt to {weights_path}")
        else:
            print(
                f"Error: outdoor_ds.ckpt not found in downloaded files. Check the Google Drive folder contents."
            )
            sys.exit(1)
    except Exception as e:
        print(f"Error downloading LoFTR weights: {e}")
        print(
            "Please download 'outdoor_ds.ckpt' manually from the Google Drive link and place it in:"
        )
        print(f"{weights_dir}")
        sys.exit(1)


class ImageListener:
    def __init__(self, text_prompt, camera="Fetch"):
        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.text_prompt = text_prompt
        self.trigger_flag = None
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.label_pub = rospy.Publisher("seg_label_refined", RosImage, queue_size=10)
        self.score_pub = rospy.Publisher("seg_score", RosImage, queue_size=10)
        self.image_pub = rospy.Publisher("seg_image", RosImage, queue_size=10)
        rospy.Subscriber(
            "/collect_mask_for_opt",
            String,
            self.trigger_for_save_pre_post_base_opt_mask_img,
        )

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

        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics

        print("\n=================================================================\n")
        print(f"Camera intrinsics\n {np.array(intrinsics)}")
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

            transform_laser = self.tf_buffer.lookup_transform(
                self.base_frame,
                "laser_link",
                rospy.Time(0),
                timeout=rospy.Duration(1.0),
            )
            trans_l = [
                transform_laser.transform.translation.x,
                transform_laser.transform.translation.y,
                transform_laser.transform.translation.z,
            ]
            rot_l = [
                transform_laser.transform.rotation.x,
                transform_laser.transform.rotation.y,
                transform_laser.transform.rotation.z,
                transform_laser.transform.rotation.w,
            ]
            RT_laser = ros_qt_to_rt(rot_l, trans_l)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            RT_camera = None
            RT_laser = None

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

        im = ros_numpy.numpify(rgb) #[:, :, ::-1]  # BGR to RGB
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera
            self.RT_laser = RT_laser

    def run_network(self):
        with lock:
            if self.im is None:
                return None, None, None, None
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera.copy() if self.RT_camera is not None else None
            RT_laser = self.RT_laser.copy() if self.RT_laser is not None else None

        im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]  # RGB to BGR
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)
        w, h = im.shape[1], im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, index = filter_large_boxes(
            image_pil_bboxes, w, h, threshold=0.5
        )
        masks = masks[index]
        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]
        bbox_annotated_pil = annotate(
            overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases
        )
        im_label = np.array(bbox_annotated_pil)

        label = mask
        self.mask = mask
        label_msg = ros_numpy.msgify(RosImage, label.astype(np.uint8), "mono8")
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
        label_msg = ros_numpy.msgify(RosImage, score.astype(np.uint8), "mono8")
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = "mono8"
        self.score_pub.publish(label_msg)

        rgb_msg = ros_numpy.msgify(RosImage, im_label, "bgr8")
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

        return im_color, depth_img, mask, RT_camera

    def trigger_for_save_pre_post_base_opt_mask_img(self, msg):
        self.trigger_flag = msg.data

    def save_mask(self, filename, img, depth, mask):
        rgb_path = os.path.join("/tmp", f"{filename}_rgb.jpg")
        depth_path = os.path.join("/tmp", f"{filename}_depth.png")
        mask_path = os.path.join("/tmp", f"{filename}_mask.png")
        cv2.imwrite(depth_path, depth)
        mask[mask > 0] = 255
        cv2.imwrite(mask_path, mask)
        PILImg.fromarray(img).save(rgb_path)


def pre_post_mask_capture(listener):
    while listener.trigger_flag is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    img, depth, mask, _ = listener.run_network()
    if img is not None:
        listener.save_mask(listener.trigger_flag, img, depth, mask)
    listener.trigger_flag = None


class BundleSDFProcessor:
    def __init__(
        self, video_dir, out_folder, use_segmenter, use_gui, stride, debug_level, K
    ):
        self.video_dir = video_dir
        self.out_folder = out_folder
        self.use_segmenter = bool(use_segmenter)
        self.use_gui = bool(use_gui)
        self.stride = stride
        self.debug_level = debug_level
        self.K = K

    def configure_bundletrack(self):
        """Configure BundleTrack and return config path and dictionary."""
        os.system(f"rm -rf {self.out_folder} && mkdir -p {self.out_folder}")

        # Download LoFTR weights if missing
        weights_dir = os.path.join(bundle_sdf_dir, "BundleTrack", "LoFTR", "weights")
        download_loftr_weights(weights_dir)

        config_path = os.path.join(bundle_sdf_dir, "BundleTrack", "config_ho3d.yml")
        if not os.path.exists(config_path):
            print(f"Error: BundleTrack config file not found at {config_path}")
            raise FileNotFoundError(f"Missing {config_path}")

        try:
            with open(config_path, "r") as f:
                cfg_bundletrack = pyyaml.safe_load(f)
        except Exception as e:
            print(f"Error loading BundleTrack config: {e}")
            raise RuntimeError(f"Failed to load {config_path}: {e}")

        cfg_bundletrack["SPDLOG"] = int(self.debug_level)
        cfg_bundletrack["depth_processing"]["percentile"] = 95
        cfg_bundletrack["erode_mask"] = 3
        cfg_bundletrack["debug_dir"] = f"{self.out_folder}/"
        cfg_bundletrack["bundle"]["max_BA_frames"] = 10
        cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
        cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
        cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
        cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
        cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
        cfg_bundletrack["feature_corres"]["map_points"] = True
        cfg_bundletrack["feature_corres"]["resize"] = 400
        cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
        cfg_bundletrack["keyframe"]["min_rot"] = 5
        cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
        cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
        cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
        cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
        cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
        cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
        cfg_bundletrack["p2p"]["max_dist"] = 0.02
        cfg_bundletrack["p2p"]["max_normal_angle"] = 45
        cfg_track_dir = os.path.join(self.out_folder, "config_bundletrack.yml")
        try:
            with open(cfg_track_dir, "w") as f:
                pyyaml.safe_dump(cfg_bundletrack, f)
        except Exception as e:
            print(f"Error writing BundleTrack config: {e}")
            raise RuntimeError(f"Failed to write {cfg_track_dir}: {e}")

        return cfg_track_dir, cfg_bundletrack

    def configure_nerf(self, cfg_bundletrack):
        """Configure NeRF and return config path and dictionary."""
        config_path = os.path.join(bundle_sdf_dir, "config.yml")
        if not os.path.exists(config_path):
            print(f"Error: NeRF config file not found at {config_path}")
            raise FileNotFoundError(f"Missing {config_path}")

        try:
            with open(config_path, "r") as f:
                cfg_nerf = pyyaml.safe_load(f)
        except Exception as e:
            print(f"Error loading NeRF config: {e}")
            raise RuntimeError(f"Failed to load {config_path}: {e}")

        cfg_nerf["continual"] = True
        cfg_nerf["trunc_start"] = 0.01
        cfg_nerf["trunc"] = 0.01
        cfg_nerf["mesh_resolution"] = 0.005
        cfg_nerf["down_scale_ratio"] = 1
        cfg_nerf["fs_sdf"] = 0.1
        cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
        cfg_nerf["datadir"] = (
            f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
        )
        cfg_nerf["notes"] = ""
        cfg_nerf["expname"] = "nerf_with_bundletrack_online"
        cfg_nerf["save_dir"] = cfg_nerf["datadir"]
        cfg_nerf_dir = os.path.join(self.out_folder, "config_nerf.yml")
        try:
            with open(cfg_nerf_dir, "w") as f:
                pyyaml.safe_dump(cfg_nerf, f)
        except Exception as e:
            print(f"Error writing NeRF config: {e}")
            raise RuntimeError(f"Failed to write {cfg_nerf_dir}: {e}")

        return cfg_nerf_dir, cfg_nerf

    def keep_relevant_files(self):
        keep = ("ob_in_cam", "cam_K.txt")
        for item in os.listdir(self.out_folder):
            item_path = os.path.join(self.out_folder, item)
            if item not in keep:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    def process(self):
        """Process video with BundleSDF."""
        try:
            cfg_track_dir, cfg_bundletrack = self.configure_bundletrack()
        except Exception as e:
            print(f"Error configuring BundleTrack: {e}")
            return

        try:
            cfg_nerf_dir, cfg_nerf = self.configure_nerf(cfg_bundletrack)
        except Exception as e:
            print(f"Error configuring NeRF: {e}")
            return

        segmenter = Segmenter() if self.use_segmenter else None

        try:
            tracker = BundleSdf(
                cfg_track_dir=cfg_track_dir,
                cfg_nerf_dir=cfg_nerf_dir,
                start_nerf_keyframes=5,
                use_gui=self.use_gui,
            )
            reader = YcbineoatReader(video_dir=self.video_dir, shorter_side=480)

            is_first_frame = True
            ob_in_cam = np.eye(4)
            ob_in_cam_dir = f"{self.out_folder}/ob_in_cam"
            os.makedirs(ob_in_cam_dir, exist_ok=True)

            for i in range(0, len(reader.color_files), self.stride):
                if rospy.is_shutdown():
                    break
                color_file = reader.color_files[i]
                color = cv2.imread(color_file)
                H0, W0 = color.shape[:2]
                depth = reader.get_depth(i)
                H, W = depth.shape[:2]
                color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

                if i == 0:
                    mask = reader.get_mask(0)
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    if segmenter:
                        mask = segmenter.run(color_file.replace("rgb", "masks"))
                else:
                    if segmenter:
                        mask = segmenter.run(color_file.replace("rgb", "masks"))
                    else:
                        mask = reader.get_mask(i)
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                if cfg_bundletrack["erode_mask"] > 0:
                    kernel = np.ones(
                        (cfg_bundletrack["erode_mask"], cfg_bundletrack["erode_mask"]),
                        np.uint8,
                    )
                    mask = cv2.erode(mask.astype(np.uint8), kernel)

                id_str = reader.id_strs[i]
                _filename, ob_in_cam, _frames = tracker.run(
                    color,
                    depth,
                    self.K,
                    id_str,
                    mask=mask,
                    occ_mask=None,
                    pose_in_model=ob_in_cam,
                )
                print(
                    "\n=================================================================\n"
                )
                print(ob_in_cam)
                print(
                    "\n=================================================================\n"
                )

                # Publish _frames images
                if _frames is not None:
                    try:
                        # Convert PIL images to NumPy arrays and remove alpha channel
                        if is_first_frame:
                            row0_rgb = np.array(_frames["row0"]["rgb"])[..., :3]
                            row0_masked_rgb = np.array(_frames["row0"]["masked_rgb"])[
                                ..., :3
                            ]
                            first_frame_pose = ob_in_cam
                            is_first_frame = False
                        row1_rgb = np.array(_frames["row1"]["rgb"])[..., :3]
                        row1_masked_rgb = np.array(_frames["row1"]["masked_rgb"])[
                            ..., :3
                        ]

                        # Convert to ROS Image messages
                        ros_row0_rgb = rnp.msgify(RosImage, row0_rgb, "rgb8")
                        ros_row0_masked_rgb = rnp.msgify(
                            RosImage, row0_masked_rgb, "rgb8"
                        )
                        ros_row1_rgb = rnp.msgify(RosImage, row1_rgb, "rgb8")
                        ros_row1_masked_rgb = rnp.msgify(
                            RosImage, row1_masked_rgb, "rgb8"
                        )

                        # Set timestamps and publish
                        stamp = rospy.Time.now()
                        ros_row0_rgb.header.stamp = stamp
                        ros_row0_rgb.header.frame_id = "camera_frame"
                        ros_row0_masked_rgb.header.stamp = stamp
                        ros_row0_masked_rgb.header.frame_id = "camera_frame"
                        ros_row1_rgb.header.stamp = stamp
                        ros_row1_rgb.header.frame_id = "camera_frame"
                        ros_row1_masked_rgb.header.stamp = stamp
                        ros_row1_masked_rgb.header.frame_id = "camera_frame"

                        pub_row0_rgb.publish(ros_row0_rgb)
                        pub_row0_masked_rgb.publish(ros_row0_masked_rgb)
                        pub_row1_rgb.publish(ros_row1_rgb)
                        pub_row1_masked_rgb.publish(ros_row1_masked_rgb)
                        rospy.loginfo(f"Published images for frame {id_str}")
                    except Exception as e:
                        rospy.logerr(f"Error publishing images: {e}")

                posetxt_filename = f"{ob_in_cam_dir}/{_filename.split('.')[0]}.txt"
                np.savetxt(posetxt_filename, ob_in_cam)

            #  publish multi array
            from std_msgs.msg import Float64MultiArray

            pub = rospy.Publisher("/bundleSDF/poses", Float64MultiArray, queue_size=10)
            msg = Float64MultiArray()
            msg.data = np.concatenate(
                (first_frame_pose.flatten(), ob_in_cam.flatten())
            ).tolist()

            # publishing 5 times just for subscribing reliably
            for _ in range(5):
                rospy.loginfo("Publishing...")
                pub.publish(msg)
                rospy.sleep(0.1)  # Small delay between publishes

            tracker.on_finish()

            # Clean up temporary files
            self.keep_relevant_files()

        except Exception as e:
            print(f"Error in BundleSDF tracking: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="run_video",
        help="run_video/global_refine/draw_pose",
    )
    parser.add_argument(
        "--task_dir", type=str, required=True, help="Directory containing task data"
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help="Text prompt for object detection",
    )
    parser.add_argument(
        "--video_dir", type=str, default="realworld", help="Directory for video data"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="realworld/out/bundlesdf",
        help="Output folder",
    )
    parser.add_argument(
        "--use_segmenter", type=int, default=0, help="Use segmenter (0 or 1)"
    )
    parser.add_argument("--use_gui", type=int, default=0, help="Use GUI (0 or 1)")
    parser.add_argument("--stride", type=int, default=1, help="Frame interval")
    parser.add_argument("--debug_level", type=int, default=0, help="Debug level")
    parser.add_argument(
        "--src_frames", type=int, default=15, help="Number of frames to collect"
    )
    parser.add_argument(
        "--realtime_frames", type=int, default=5, help="Number of frames to collect"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Fetch",
        help="Camera type (Fetch, Realsense, Azure, etc.)",
    )
    args = parser.parse_args()

    rospy.init_node("merged_bundlesdf_node", anonymous=True)
    pub_row0_rgb = rospy.Publisher(
        "/bundleSDF/src_rgb_with_pose", RosImage, queue_size=10
    )
    pub_row0_masked_rgb = rospy.Publisher(
        "/bundleSDF/src_masked_rgb", RosImage, queue_size=10
    )
    pub_row1_rgb = rospy.Publisher(
        "/bundleSDF/dst_rgb_with_pose", RosImage, queue_size=10
    )
    pub_row1_masked_rgb = rospy.Publisher(
        "/bundleSDF/dst_masked_rgb", RosImage, queue_size=10
    )

    listener = ImageListener(text_prompt=args.text_prompt, camera=args.camera)
    rospy.sleep(0.1)

    root_dir = os.path.dirname(os.path.normpath(args.task_dir))
    realworld_dir = args.video_dir
    rgb_source_dir = os.path.join(args.task_dir, "rgb")
    depth_source_dir = os.path.join(args.task_dir, "depth")
    pose_source_dir = os.path.join(args.task_dir, "pose")
    sam2_out_dir = os.path.join(args.task_dir, "out", "samv2")
    try:
        mask_source_dir = os.path.join(
            sam2_out_dir, os.listdir(sam2_out_dir)[-1], "obj_masks"
        )
    except IndexError:
        print(f"Error: No SAMv2 output directories found in {sam2_out_dir}")
        sys.exit(1)
    realworld_rgb_dir = os.path.join(realworld_dir, "rgb")
    realworld_depth_dir = os.path.join(realworld_dir, "depth")
    realworld_pose_dir = os.path.join(realworld_dir, "pose")
    realworld_mask_dir = os.path.join(realworld_dir, "masks")
    cam_k_file_source = os.path.join(root_dir, "cam_K.txt")
    cam_k_file_target = os.path.join(realworld_dir, "cam_K.txt")

    os.makedirs(realworld_rgb_dir, exist_ok=True)
    os.makedirs(realworld_depth_dir, exist_ok=True)
    os.makedirs(realworld_pose_dir, exist_ok=True)
    os.makedirs(realworld_mask_dir, exist_ok=True)

    # Loop over human demo frames
    rospy.loginfo(f"Collecting {args.src_frames} frames from source directories...")

    last = len(os.listdir(rgb_source_dir))-1
    for i in range(args.src_frames):
        # frame_str = f"{i:06d}"
        frame_str = f"{last:06d}"

        files = [
            (rgb_source_dir, realworld_rgb_dir, f"{frame_str}.jpg", "RGB"),
            (depth_source_dir, realworld_depth_dir, f"{frame_str}.png", "Depth"),
            (mask_source_dir, realworld_mask_dir, f"{frame_str}.png", "Mask"),
            (pose_source_dir, realworld_pose_dir, f"{frame_str}.npz", "Pose"),
        ]

        for src_dir, tgt_dir, filename, label in files:
            src_file = os.path.join(src_dir, filename)
            tgt_file = os.path.join(tgt_dir, filename)

            if os.path.exists(src_file):
                shutil.copy(src_file, tgt_file)
                # print(f"Copied {label} {src_file} to {tgt_file}")
            else:
                print(f"{label} for frame {src_file} does not exist.")

        if os.path.exists(cam_k_file_source):
            shutil.copy(cam_k_file_source, cam_k_file_target)
            print(f"Copied {cam_k_file_source} to {cam_k_file_target}")
        else:
            print(f"{cam_k_file_source} does not exist.")

    print("Step-2: Get real-time RGB, depth, masks from robot")
    curr_frame = last + args.src_frames
    print(f"Collecting real-time RGBD+SAM-mask for {args.realtime_frames} frames...")
    time.sleep(0.1)
    while (
        not rospy.is_shutdown() and curr_frame <= last + args.src_frames + args.realtime_frames
    ):
        print(curr_frame)
        try:
            img, depth, mask, RT_camera = listener.run_network()
            print(f"{len(np.unique(mask)) - 1} objects detected")

            if img is None:
                continue

            rgb_path = os.path.join(realworld_rgb_dir, f"{curr_frame:06d}.jpg")
            depth_path = os.path.join(realworld_depth_dir, f"{curr_frame:06d}.png")
            mask_path = os.path.join(realworld_mask_dir, f"{curr_frame:06d}.png")
            pose_path = os.path.join(realworld_pose_dir, f"{curr_frame:06d}.npz")

            cv2.imwrite(depth_path, depth)
            mask[mask > 0] = 255
            cv2.imwrite(mask_path, mask)
            PILImg.fromarray(img).save(rgb_path)
            np.savez(pose_path, RT_camera=RT_camera)

            curr_frame += 1
        except Exception as e:
            print(f"Error in frame {curr_frame}: {e}")
            continue

    print("Running BundleSDF...")
    time.sleep(0.1)
    K = listener.intrinsics
    processor = BundleSDFProcessor(
        video_dir=args.video_dir,
        out_folder=args.out_folder,
        use_segmenter=args.use_segmenter,
        use_gui=args.use_gui,
        stride=args.stride,
        debug_level=args.debug_level,
        K=K,
    )
    try:
        processor.process()
    except Exception as e:
        print(f"Error in BundleSDF processing: {e}")
