# my_ros_package/scripts/config_manager.py
import yaml
import os

class ConfigManager:
    def __init__(self, code_dir, out_folder, debug_level):
        self.code_dir = code_dir
        self.out_folder = out_folder
        self.debug_level = debug_level

    def load_bundletrack_config(self):
        cfg = yaml.load(open(f"{self.code_dir}/BundleTrack/config_ho3d.yml", "r"), Loader=yaml.SafeLoader)
        cfg["SPDLOG"] = int(self.debug_level)
        cfg["depth_processing"]["percentile"] = 95
        cfg["erode_mask"] = 3
        cfg["debug_dir"] = self.out_folder + "/"
        cfg["bundle"]["max_BA_frames"] = 10
        cfg["bundle"]["max_optimized_feature_loss"] = 0.03
        cfg["feature_corres"]["max_dist_neighbor"] = 0.02
        cfg["feature_corres"]["max_normal_neighbor"] = 30
        cfg["feature_corres"]["max_dist_no_neighbor"] = 0.01
        cfg["feature_corres"]["max_normal_no_neighbor"] = 20
        cfg["feature_corres"]["map_points"] = True
        cfg["feature_corres"]["resize"] = 400
        cfg["feature_corres"]["rematch_after_nerf"] = True
        cfg["keyframe"]["min_rot"] = 5
        cfg["ransac"]["inlier_dist"] = 0.01
        cfg["ransac"]["inlier_normal_angle"] = 20
        cfg["ransac"]["max_trans_neighbor"] = 0.02
        cfg["ransac"]["max_rot_deg_neighbor"] = 30
        cfg["ransac"]["max_trans_no_neighbor"] = 0.01
        cfg["ransac"]["max_rot_no_neighbor"] = 10
        cfg["p2p"]["max_dist"] = 0.02
        cfg["p2p"]["max_normal_angle"] = 45
        cfg_track_dir = f"{self.out_folder}/config_bundletrack.yml"
        yaml.dump(cfg, open(cfg_track_dir, "w"))
        return cfg, cfg_track_dir

    def load_nerf_config(self, bundletrack_cfg):
        cfg = yaml.load(open(f"{self.code_dir}/config.yml", "r"), Loader=yaml.SafeLoader)
        cfg["continual"] = True
        cfg["trunc_start"] = 0.01
        cfg["trunc"] = 0.01
        cfg["mesh_resolution"] = 0.005
        cfg["down_scale_ratio"] = 1
        cfg["fs_sdf"] = 0.1
        cfg["far"] = bundletrack_cfg["depth_processing"]["zfar"]
        cfg["datadir"] = f"{bundletrack_cfg['debug_dir']}/nerf_with_bundletrack_online"
        cfg["notes"] = ""
        cfg["expname"] = "nerf_with_bundletrack_online"
        cfg["save_dir"] = cfg["datadir"]
        cfg_nerf_dir = f"{self.out_folder}/config_nerf.yml"
        yaml.dump(cfg, open(cfg_nerf_dir, "w"))
        return cfg, cfg_nerf_dir