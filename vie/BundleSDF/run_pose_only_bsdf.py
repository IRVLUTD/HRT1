# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Modified for HRT1 by Jishnu Jaykumar Padalunkal

"""
removed 3D reconstruction
contains only feature correspondence and pose estimation
"""

from bundlesdf import *
import argparse
import os, sys

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
# from segmentation_utils import Segmenter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../realworld")
    parser.add_argument("--out_folder", type=str, default="../realworld/out/bundlesdf")
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="interval of frames to run; 1 means using every frame",
    )
    parser.add_argument(
        "--debug_level", type=int, default=0, help="higher means more logging"
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    out_folder = args.out_folder
    use_segmenter = args.use_segmenter
    use_gui = args.use_gui

    set_seed(0)

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")

    cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml", "r"))
    cfg_bundletrack["SPDLOG"] = int(args.debug_level)
    # cfg_bundletrack['depth_processing']["zfar"] = 1
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
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
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", "r"))
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    # if use_segmenter:
    #   segmenter = Segmenter()

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=use_gui,
    )

    reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)
    _pose_in_model = np.eye(4)

    K = reader.K.copy()
    print(f"camera intrinsics: {K}")

    for i in range(0, len(reader.color_files), args.stride):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        H0, W0 = color.shape[:2]
        depth = reader.get_depth(i)  # * 1000
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if i == 0:
            mask = reader.get_mask(0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
        else:
            if use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
            else:
                mask = reader.get_mask(i)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg_bundletrack["erode_mask"] > 0:
            kernel = np.ones(
                (cfg_bundletrack["erode_mask"], cfg_bundletrack["erode_mask"]), np.uint8
            )
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        id_str = reader.id_strs[i]
        _filename, _pose_in_model, _frames = tracker.run(
            color,
            depth,
            K,
            id_str,
            mask=mask,
            occ_mask=None,
            pose_in_model=_pose_in_model,
        )

        print(_pose_in_model)
    tracker.on_finish()
