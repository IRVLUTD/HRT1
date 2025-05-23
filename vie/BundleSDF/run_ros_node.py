#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Modified for HRT1 by Jishnu Jaykumar Padalunkal

import rospy
import os
import sys
import argparse
import numpy as np
from std_srvs.srv import Trigger
from bundlesdf import BundleSdf, set_seed
from my_utils.config_manager import ConfigManager
from my_utils.frame_processor import FrameProcessor
from my_utils.image_publisher import ImagePublisher

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../realworld")
    parser.add_argument("--out_folder", type=str, default="../realworld/out/bundlesdf")
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="interval of frames to run; 1 means using every frame")
    parser.add_argument("--debug_level", type=int, default=0, help="higher means more logging")
    args = parser.parse_args()

    rospy.init_node("bundlesdf_node", anonymous=True)

    # Wait for data_collection_node to complete
    wait_service = rospy.get_param("~wait_for_service", "")
    if wait_service:
        rospy.loginfo(f"Waiting for service {wait_service}")
        rospy.wait_for_service(wait_service)
        try:
            ready = rospy.ServiceProxy(wait_service, Trigger)
            response = ready()
            if response.success:
                rospy.loginfo("Data collection complete, starting BundleSDF")
            else:
                rospy.logwarn(f"Data collection not ready: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    set_seed(0)
    os.system(f"rm -rf {args.out_folder} && mkdir -p {args.out_folder}")
    config_manager = ConfigManager(code_dir, args.out_folder, args.debug_level)
    bundletrack_cfg, cfg_track_dir = config_manager.load_bundletrack_config()
    nerf_cfg, cfg_nerf_dir = config_manager.load_nerf_config(bundletrack_cfg)
    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=args.use_gui,
    )
    frame_processor = FrameProcessor(
        video_dir=args.video_dir,
        shorter_side=480,
        use_segmenter=args.use_segmenter,
        erode_mask_size=bundletrack_cfg["erode_mask"],
    )
    image_publisher = ImagePublisher()
    K = frame_processor.get_intrinsics()
    print(f"camera intrinsics: {K}")
    pose_in_model = np.eye(4)
    for i in range(0, frame_processor.get_total_frames(), args.stride):
        if rospy.is_shutdown():
            break
        color, depth, mask, id_str = frame_processor.process_frame(i, args.stride)
        if color is None:
            break
        filename, pose_in_model, frames = tracker.run(
            color,
            depth,
            K,
            id_str,
            mask=mask,
            occ_mask=None,
            pose_in_model=pose_in_model,
        )
        image_publisher.publish_frames(frames, id_str)
        print(pose_in_model)
    tracker.on_finish()

if __name__ == "__main__":
    main()