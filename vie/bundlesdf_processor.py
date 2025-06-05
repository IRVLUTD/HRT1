#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

import os
import sys
import cv2
import time
import rospy
import shutil
import numpy as np
from PIL import Image as PILImg
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image as RosImage
import ros_numpy as rnp

# Add BundleSDF directory to sys.path
code_dir = os.path.dirname(os.path.realpath(__file__))
bundle_sdf_dir = os.path.join(code_dir, "BundleSDF")
sys.path.append(bundle_sdf_dir)
try:
    from bundlesdf import BundleSdf, YcbineoatReader
    from segmentation_utils import Segmenter
except ModuleNotFoundError as e:
    print(f"Error importing BundleSDF modules: {e}")
    print(
        "Ensure 'my_cpp' module is built in BundleSDF/build/. Run 'cmake .. && make' in BundleSDF/build/"
    )
    sys.exit(1)
from utils import download_loftr_weights, remove_unecessary_files
import yaml as pyyaml


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
        self.bundle_sdf_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "BundleSDF"
        )

    def configure_bundletrack(self):
        os.system(f"rm -rf {self.out_folder} && mkdir -p {self.out_folder}")
        weights_dir = os.path.join(
            self.bundle_sdf_dir, "BundleTrack", "LoFTR", "weights"
        )
        download_loftr_weights(weights_dir)
        config_path = os.path.join(
            self.bundle_sdf_dir, "BundleTrack", "config_ho3d.yml"
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing {config_path}")
        with open(config_path, "r") as f:
            cfg_bundletrack = pyyaml.safe_load(f)
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
        with open(cfg_track_dir, "w") as f:
            pyyaml.safe_dump(cfg_bundletrack, f)
        return cfg_track_dir, cfg_bundletrack

    def configure_nerf(self, cfg_bundletrack):
        config_path = os.path.join(self.bundle_sdf_dir, "config.yml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing {config_path}")
        with open(config_path, "r") as f:
            cfg_nerf = pyyaml.safe_load(f)
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
        with open(cfg_nerf_dir, "w") as f:
            pyyaml.safe_dump(cfg_nerf, f)
        return cfg_nerf_dir, cfg_nerf

    def keep_relevant_files(self):
        keep = ("ob_in_cam", "pose_overlayed_rgb", "cam_K.txt")
        for item in os.listdir(self.out_folder):
            item_path = os.path.join(self.out_folder, item)
            if item not in keep:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    def process(
        self, pub_row0_rgb, pub_row0_masked_rgb, pub_row1_rgb, pub_row1_masked_rgb
    ):
        start_time = time.time()
        cfg_track_dir, cfg_bundletrack = self.configure_bundletrack()
        cfg_nerf_dir, cfg_nerf = self.configure_nerf(cfg_bundletrack)
        segmenter = Segmenter() if self.use_segmenter else None
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
        pose_overlayed_rgb_dir = f"{self.out_folder}/pose_overlayed_rgb"
        os.makedirs(ob_in_cam_dir, exist_ok=True)
        os.makedirs(pose_overlayed_rgb_dir, exist_ok=True)

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
            # print("\n=================================================================\n")
            # print(ob_in_cam)
            # print("\n=================================================================\n")
            if _frames is not None:
                try:
                    if is_first_frame:
                        row0_rgb = np.array(_frames["row0"]["rgb"])[..., :3]
                        # row0_masked_rgb = np.array(_frames["row0"]["masked_rgb"])[..., :3]
                        first_frame_pose = ob_in_cam
                        is_first_frame = False
                    row1_rgb = np.array(_frames["row1"]["rgb"])[..., :3]
                    # row1_masked_rgb = np.array(_frames["row1"]["masked_rgb"])[..., :3]
                    ros_row0_rgb = rnp.msgify(RosImage, row0_rgb, "rgb8")
                    # ros_row0_masked_rgb = rnp.msgify(RosImage, row0_masked_rgb, "rgb8")
                    ros_row1_rgb = rnp.msgify(RosImage, row1_rgb, "rgb8")
                    # ros_row1_masked_rgb = rnp.msgify(RosImage, row1_masked_rgb, "rgb8")
                    stamp = rospy.Time.now()
                    ros_row0_rgb.header.stamp = stamp
                    ros_row0_rgb.header.frame_id = "camera_frame"
                    # ros_row0_masked_rgb.header.stamp = stamp
                    # ros_row0_masked_rgb.header.frame_id = "camera_frame"
                    ros_row1_rgb.header.stamp = stamp
                    ros_row1_rgb.header.frame_id = "camera_frame"
                    # ros_row1_masked_rgb.header.stamp = stamp
                    # ros_row1_masked_rgb.header.frame_id = "camera_frame"
                    pub_row0_rgb.publish(ros_row0_rgb)
                    # pub_row0_masked_rgb.publish(ros_row0_masked_rgb)
                    pub_row1_rgb.publish(ros_row1_rgb)
                    # pub_row1_masked_rgb.publish(ros_row1_masked_rgb)
                    rospy.loginfo(f"Published images for frame {id_str}")
                except Exception as e:
                    rospy.logerr(f"Error publishing images: {e}")
            _filename = _filename.split(".")[0]
            posetxt_filename = f"{ob_in_cam_dir}/{_filename}.txt"
            np.savetxt(posetxt_filename, ob_in_cam)
            PILImg.fromarray(row1_rgb).save(
                os.path.join(pose_overlayed_rgb_dir, f"{_filename}.png")
            )

        # cue (jishnu): uncomment to publish the first frame pose and ob_in_cam
        # pub = rospy.Publisher("/bundleSDF/poses", Float64MultiArray, queue_size=10)
        # msg = Float64MultiArray()
        # msg.data = np.concatenate((first_frame_pose.flatten(), ob_in_cam.flatten())).tolist()
        # for _ in range(5):
        #     rospy.loginfo("Publishing...")
        #     pub.publish(msg)
        #     rospy.sleep(0.1)

        remove_unecessary_files(self.out_folder)
        tracker.on_finish()
        self.keep_relevant_files()
        process_time = time.time() - start_time
        return process_time
