import numpy as np
import cv2
import yaml
import os
import sys
import importlib.util
import logging

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    import matplotlib.pyplot as plt
from loftr_wrapper import LoftrRunner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths (flexible for local setup)
CODE_DIR = "./"

# DATA_ROOT = "/home/jishnu/Projects/mm-demo/vie/for-paper-sub/bundleSDF-rw-stanford/realworld_chair"
DATA_ROOT = "/home/jishnu/Projects/mm-demo/vie/data/test_obj_pose"

BUILD_DIR = os.path.abspath(f"{CODE_DIR}/BundleTrack/build")
sys.path.append(BUILD_DIR)

# Debug: Verify module path
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for my_cpp in: {BUILD_DIR}")
print(f"Build directory contents: {os.listdir(BUILD_DIR)}")
print(f"sys.path: {sys.path}")

# Explicitly load my_cpp module
my_cpp_path = os.path.join(BUILD_DIR, "my_cpp.cpython-38-x86_64-linux-gnu.so")
if not os.path.exists(my_cpp_path):
    raise FileNotFoundError(f"my_cpp module not found at {my_cpp_path}")
spec = importlib.util.spec_from_file_location("my_cpp", my_cpp_path)
if spec is None:
    raise ImportError(f"Failed to create spec for my_cpp at {my_cpp_path}")
my_cpp = importlib.util.module_from_spec(spec)
sys.modules["my_cpp"] = my_cpp
try:
    spec.loader.exec_module(my_cpp)
    print("Successfully loaded my_cpp module")
    print(f"my_cpp contents: {dir(my_cpp)}")
except Exception as e:
    print(f"Error loading my_cpp module: {e}")
    raise

def transform_pts(pts, tf):
    """Transform 2D points using a transformation matrix."""
    try:
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_transformed = (tf @ pts_h.T).T
        return pts_transformed[:, :2] / pts_transformed[:, 2:3]
    except Exception as e:
        logging.error(f"Error in transform_pts: {e}")
        return None

def get_3d_points(matches, depth1, depth2, K):
    """Convert 2D correspondences to 3D points using depth and intrinsics."""
    try:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u1, v1 = matches[:, 0].astype(int), matches[:, 1].astype(int)
        u2, v2 = matches[:, 2].astype(int), matches[:, 3].astype(int)

        z1 = depth1[v1, u1]
        z2 = depth2[v2, u2]
        valid = (z1 > 0) & (z2 > 0)
        u1, v1, u2, v2, z1, z2 = u1[valid], v1[valid], u2[valid], v2[valid], z1[valid], z2[valid]

        if len(u1) == 0:
            logging.warning("No valid 3D correspondences after depth filtering")
            return None, None

        pts1_3d = np.zeros((len(u1), 3))
        pts2_3d = np.zeros((len(u2), 3))
        pts1_3d[:, 0] = (u1 - cx) * z1 / fx
        pts1_3d[:, 1] = (v1 - cy) * z1 / fy
        pts1_3d[:, 2] = z1
        pts2_3d[:, 0] = (u2 - cx) * z2 / fx
        pts2_3d[:, 1] = (v2 - cy) * z2 / fy
        pts2_3d[:, 2] = z2

        logging.info(f"Generated {len(pts1_3d)} 3D correspondence points")
        return pts1_3d, pts2_3d
    except Exception as e:
        logging.error(f"Error in get_3d_points: {e}")
        return None, None

def visualize_point_clouds(pcd1, pcd2, rgb1, rgb2, corres1, corres2, relative_pose):
    """Visualize registered point clouds with RGB colors and correspondences."""
    try:
        if OPEN3D_AVAILABLE:
            pcd1_o3d = o3d.geometry.PointCloud()
            pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)
            pcd1_o3d.colors = o3d.utility.Vector3dVector(rgb1 / 255.0)  # Normalize to [0, 1]

            pcd2_o3d = o3d.geometry.PointCloud()
            pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)
            pcd2_o3d.colors = o3d.utility.Vector3dVector(rgb2 / 255.0)
            pcd2_o3d.transform(relative_pose)  # Align to frame 1

            geometries = [pcd1_o3d, pcd2_o3d]
            if corres1 is not None and corres2 is not None:
                corres1_o3d = o3d.geometry.PointCloud()
                corres1_o3d.points = o3d.utility.Vector3dVector(corres1)
                corres1_o3d.paint_uniform_color([1, 1, 0])  # Yellow for frame 1

                corres2_o3d = o3d.geometry.PointCloud()
                corres2_o3d.points = o3d.utility.Vector3dVector(corres2)
                corres2_o3d.transform(relative_pose)
                corres2_o3d.paint_uniform_color([0, 1, 1])  # Cyan for frame 2

                geometries.extend([corres1_o3d, corres2_o3d])

            o3d.visualization.draw_geometries(geometries)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], c=rgb1 / 255.0, label='Frame 1', s=2)
            
            pcd2_h = np.hstack([pcd2, np.ones((pcd2.shape[0], 1))])
            pcd2_transformed = (relative_pose @ pcd2_h.T).T[:, :3]
            ax.scatter(pcd2_transformed[:, 0], pcd2_transformed[:, 1], pcd2_transformed[:, 2], c=rgb2 / 255.0, label='Frame 2', s=2)

            if corres1 is not None and corres2 is not None:
                ax.scatter(corres1[:, 0], corres1[:, 1], corres1[:, 2], c='y', label='Corres 1', s=20)
                corres2_h = np.hstack([corres2, np.ones((corres2.shape[0], 1))])
                corres2_transformed = (relative_pose @ corres2_h.T).T[:, :3]
                ax.scatter(corres2_transformed[:, 0], corres2_transformed[:, 1], corres2_transformed[:, 2], c='c', label='Corres 2', s=20)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.legend()
            ax.set_title('Registered Point Clouds with RGB Colors and Correspondences')
            plt.tight_layout()
            output_path = os.path.join(CODE_DIR, 'debug', 'point_clouds_rgb.png')
            plt.savefig(output_path, dpi=300)
            logging.info(f"Saved visualization to {output_path}")
            plt.close()
    except Exception as e:
        logging.error(f"Error in visualize_point_clouds: {e}")

class BundleSdf:
    def __init__(self, cfg_track_dir=f"{CODE_DIR}/BundleTrack/config_ho3d.yml"):
        cfg_track_dir = os.path.abspath(cfg_track_dir)
        if not os.path.exists(cfg_track_dir):
            raise FileNotFoundError(f"Config file not found: {cfg_track_dir}")
        with open(cfg_track_dir, 'r') as ff:
            self.cfg_track = yaml.load(ff, Loader=yaml.FullLoader)
        self.debug_dir = os.path.join(CODE_DIR, "debug")
        self.SPDLOG = self.cfg_track.get("SPDLOG", 2)
        
        try:
            self.loftr = LoftrRunner()
        except Exception as e:
            print(f"Error initializing LoFTR: {e}")
            raise
        
        try:
            yml = my_cpp.YamlLoadFile(cfg_track_dir)
            self.bundler = my_cpp.Bundler(yml)
        except AttributeError as e:
            print(f"Error: my_cpp does not have Bundler: {e}")
            raise
        except Exception as e:
            print(f"Error initializing Bundler: {e}")
            raise
        
        self.cnt = -1
        self.K = None
        self.last_matches = None

    def make_frame(self, color, depth, K, id_str, mask=None, pose_in_model=np.eye(4)):
        try:
            H, W = color.shape[:2]
            roi = [0, W-1, 0, H-1]
            frame = my_cpp.Frame(color, depth, roi, pose_in_model, self.cnt, id_str, K, self.bundler.yml)
            if mask is not None:
                frame._fg_mask = my_cpp.cvMat(mask)
            return frame
        except Exception as e:
            logging.error(f"Error in make_frame: {e}")
            return None

    def find_corres(self, frame_pairs):
        logging.info(f"frame_pairs: {len(frame_pairs)}")
        try:
            is_match_ref = len(frame_pairs) == 1 and frame_pairs[0][0]._ref_frame_id == frame_pairs[0][1]._id and self.bundler._newframe == frame_pairs[0][0]

            imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
            imgs = np.array([np.array(img) for img in imgs])

            if len(query_pairs) == 0:
                logging.warning("No query pairs for correspondence finding")
                return

            corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
            for i_pair in range(len(query_pairs)):
                cur_corres = corres[i_pair][:, :4]
                logging.info(f"LoFTR matches for pair {i_pair}: {len(cur_corres)}")
                tfA = np.array(tfs[i_pair * 2])
                tfB = np.array(tfs[i_pair * 2 + 1])
                cur_corres[:, :2] = transform_pts(cur_corres[:, :2], np.linalg.inv(tfA))
                cur_corres[:, 2:4] = transform_pts(cur_corres[:, 2:4], np.linalg.inv(tfB))
                self.bundler._fm._raw_matches[query_pairs[i_pair]] = cur_corres.round().astype(np.uint16)

            min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

            if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]]) < min_match_with_ref:
                self.bundler._fm._raw_matches[frame_pairs[0]] = []
                self.bundler._newframe._status = my_cpp.Frame.FAIL
                logging.info(f'frame {self.bundler._newframe._id_str} mark FAIL, due to no matching')
                return

            self.bundler._fm.rawMatchesToCorres(query_pairs)
            self.bundler._fm.runRansacMultiPairGPU(query_pairs)

            # Debug: Inspect MapCorrespondences
            logging.info(f"MapCorrespondences methods: {dir(self.bundler._fm._matches)}")
            if frame_pairs:
                try:
                    self.last_matches = self.bundler._fm._matches[frame_pairs[0]]
                    logging.info(f"Retrieved {len(self.last_matches)} matches for frame pair")
                except KeyError:
                    logging.warning(f"No matches found for frame pair {frame_pairs[0]}")
                    self.last_matches = None
        except Exception as e:
            logging.error(f"Error in find_corres: {e}")

    def process_new_frame(self, frame):
        logging.info(f"process frame {frame._id_str}")
        try:
            self.bundler._newframe = frame
            os.makedirs(self.debug_dir, exist_ok=True)

            if frame._id > 0:
                ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
                frame._ref_frame_id = ref_frame._id
                frame._pose_in_model = ref_frame._pose_in_model
            else:
                self.bundler._firstframe = frame

            frame.invalidatePixelsByMask(frame._fg_mask)
            n_fg = (np.array(frame._fg_mask) > 0).sum()
            logging.info(f"Frame {frame._id_str} foreground pixels: {n_fg}")
            if n_fg < 100:
                logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
                frame._status = my_cpp.Frame.FAIL
                self.bundler.forgetFrame(frame)
                return

            if self.cfg_track["depth_processing"]["denoise_cloud"]:
                frame.pointCloudDenoise()

            n_valid = frame.countValidPoints()
            logging.info(f"Frame {frame._id_str} valid points: {n_valid}")
            if frame._id == 0:
                self.bundler.checkAndAddKeyframe(frame)
                self.bundler._frames[frame._id] = frame
                return

            n_valid_first = self.bundler._firstframe.countValidPoints()
            if n_valid < n_valid_first / 40.0:
                logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
                frame._status = my_cpp.Frame.FAIL
                self.bundler.forgetFrame(frame)
                return

            min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
            self.find_corres([(frame, ref_frame)])
            try:
                matches = self.bundler._fm._matches[(frame, ref_frame)]
                logging.info(f"Matches for pose estimation: {len(matches)}")
            except KeyError:
                matches = []
                logging.warning(f"No matches found for frame pair {(frame, ref_frame)}")

            if frame._status == my_cpp.Frame.FAIL:
                logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
                self.bundler.forgetFrame(frame)
                return

            if len(matches) < min_match_with_ref:
                visibles = []
                for kf in self.bundler._keyframes:
                    visible = my_cpp.computeCovisibility(frame, kf)
                    visibles.append(visible)
                visibles = np.array(visibles)
                ids = np.argsort(visibles)[::-1]
                found = False
                for id in ids:
                    kf = self.bundler._keyframes[id]
                    logging.info(f"trying new ref frame {kf._id_str}")
                    ref_frame = kf
                    frame._ref_frame_id = kf._id
                    frame._pose_in_model = kf._pose_in_model
                    self.find_corres([(frame, ref_frame)])
                    try:
                        matches = self.bundler._fm._matches[(frame, kf)]
                        logging.info(f"Matches with new ref frame {kf._id_str}: {len(matches)}")
                    except KeyError:
                        matches = []
                    if len(matches) >= min_match_with_ref:
                        logging.info(f"re-choose new ref frame to {kf._id_str}")
                        found = True
                        break
                if not found:
                    frame._status = my_cpp.Frame.FAIL
                    logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
                    self.bundler.forgetFrame(frame)
                    return

            logging.info(f"frame {frame._id_str} pose update before\n{frame._pose_in_model.round(3)}")
            offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
            frame._pose_in_model = offset @ frame._pose_in_model
            logging.info(f"frame {frame._id_str} pose update after\n{frame._pose_in_model.round(3)}")

            window_size = self.cfg_track["bundle"]["window_size"]
            if len(self.bundler._frames) - len(self.bundler._keyframes) > window_size:
                for k in self.bundler._frames:
                    f = self.bundler._frames[k]
                    isforget = self.bundler.forgetFrame(f)
                    if isforget:
                        logging.info(f"exceed window size, forget frame {f._id_str}")
                        break

            self.bundler._frames[frame._id] = frame
            self.bundler.selectKeyFramesForBA()
            local_frames = self.bundler._local_frames
            pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
            self.find_corres(pairs)
            if frame._status == my_cpp.Frame.FAIL:
                self.bundler.forgetFrame(frame)
                return

            find_matches = False
            self.bundler.optimizeGPU(local_frames, find_matches)
            if frame._status == my_cpp.Frame.FAIL:
                self.bundler.forgetFrame(frame)
                return

            self.bundler.checkAndAddKeyframe(frame)
        except Exception as e:
            logging.error(f"Error in process_new_frame: {e}")

    def run(self, color, depth, K, id_str, mask=None):
        self.cnt += 1
        if self.K is None:
            self.K = K

        try:
            H, W = color.shape[:2]
            logging.info(f"Frame {id_str} depth min: {depth.min()}, max: {depth.max()}, mean: {depth[depth > 0].mean() if (depth > 0).sum() > 0 else 0}")
            logging.info(f"Frame {id_str} mask pixels: {(mask > 0).sum()}")

            # Temporarily disable aggressive denoising to avoid zero-point-cloud issue
            percentile = self.cfg_track['depth_processing'].get("percentile", 100)
            if percentile < 100:
                logging.info("percentile denoise start")
                valid = (depth >= 0.1) & (mask > 0)
                if valid.sum() == 0:
                    logging.warning(f"Frame {id_str} has no valid depth pixels after filtering")
                    return None
                thres = np.percentile(depth[valid], percentile)
                logging.info(f"Frame {id_str} depth threshold: {thres}")
                depth = depth.copy()
                depth[depth >= thres] = 0
                logging.info("percentile denoise done")

            frame = self.make_frame(color, depth, K, id_str, mask)
            if frame is None:
                return None
            os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)
            logging.info(f"processNewFrame start {frame._id_str}")
            self.process_new_frame(frame)
            logging.info(f"processNewFrame done {frame._id_str}")
            return frame._pose_in_model
        except Exception as e:
            logging.error(f"Error in run: {e}")
            return None

class PoseTracker:
    def __init__(self, config_path=f"{CODE_DIR}/BundleTrack/config_ho3d.yml"):
        self.bundle_sdf = BundleSdf(cfg_track_dir=config_path)

    def estimate_relative_pose(self, rgb1, depth1, mask1, rgb2, depth2, mask2, cam_K):
        try:
            if rgb1.shape[:2] != rgb2.shape[:2] or depth1.shape != depth2.shape:
                raise ValueError(f"RGB and depth images must have same dimensions")
            if rgb1.shape[:2] != depth1.shape:
                raise ValueError(f"RGB and depth dimensions must match")
            if mask1 is None or mask2 is None:
                raise ValueError("Masks are required")
            if len(mask1.shape) > 2:
                raise ValueError(f"Mask1 has unexpected shape {mask1.shape}, expected 2D")
            if len(mask2.shape) > 2:
                raise ValueError(f"Mask2 has unexpected shape {mask2.shape}, expected 2D")

            logging.info(f"RGB1 shape: {rgb1.shape}, Depth1 shape: {depth1.shape}, Mask1 shape: {mask1.shape}")
            logging.info(f"RGB2 shape: {rgb2.shape}, Depth2 shape: {depth2.shape}, Mask2 shape: {mask2.shape}")

            H, W = depth1.shape[:2]
            rgb1 = cv2.resize(rgb1, (W, H), interpolation=cv2.INTER_NEAREST)
            rgb2 = cv2.resize(rgb2, (W, H), interpolation=cv2.INTER_NEAREST)
            depth1 = cv2.resize(depth1, (W, H), interpolation=cv2.INTER_NEAREST)
            depth2 = cv2.resize(depth2, (W, H), interpolation=cv2.INTER_NEAREST)
            mask1 = cv2.resize(mask1, (W, H), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.resize(mask2, (W, H), interpolation=cv2.INTER_NEAREST)
            rgb1_proc = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
            rgb2_proc = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
            depth1 = depth1.astype(np.float32) / 1000.0
            depth2 = depth2.astype(np.float32) / 1000.0
            mask1 = (mask1 > 0).astype(np.uint8)
            mask2 = (mask2 > 0).astype(np.uint8)

            pose1 = self.bundle_sdf.run(rgb1_proc, depth1, cam_K, "000000", mask=mask1)
            if pose1 is None:
                logging.error("Failed to process first frame")
                return None, None, None, None, None, None, None

            pose2 = self.bundle_sdf.run(rgb2_proc, depth2, cam_K, "000001", mask=mask2)
            if pose2 is None:
                logging.error("Failed to process second frame")
                return None, None, None, None, None, None, None

            pose1_inv = np.linalg.inv(pose1)
            relative_pose = pose2 @ pose1_inv
            logging.info(f"Relative pose:\n{relative_pose.round(3)}")

            matches = self.bundle_sdf.last_matches
            corres1_3d, corres2_3d = None, None
            if matches is not None:
                corres1_3d, corres2_3d = get_3d_points(matches, depth1, depth2, cam_K)

            u, v = np.meshgrid(np.arange(W), np.arange(H))
            valid1 = (depth1 > 0) & (mask1 > 0)
            valid2 = (depth2 > 0) & (mask2 > 0)
            u1, v1 = u[valid1], v[valid1]
            z1 = depth1[valid1]
            u2, v2 = u[valid2], v[valid2]
            z2 = depth2[valid2]

            logging.info(f"Frame 1 point cloud size: {len(u1)}")
            logging.info(f"Frame 2 point cloud size: {len(u2)}")

            fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
            pcd1 = np.zeros((len(u1), 3))
            pcd1[:, 0] = (u1 - cx) * z1 / fx
            pcd1[:, 1] = (v1 - cy) * z1 / fy
            pcd1[:, 2] = z1
            pcd2 = np.zeros((len(u2), 3))
            pcd2[:, 0] = (u2 - cx) * z2 / fx
            pcd2[:, 1] = (v2 - cy) * z2 / fy
            pcd2[:, 2] = z2

            # Extract RGB colors for valid points
            rgb1_colors = rgb1_proc[v1.astype(int), u1.astype(int)]  # Shape: (N, 3)
            rgb2_colors = rgb2_proc[v2.astype(int), u2.astype(int)]  # Shape: (M, 3)

            return relative_pose, pcd1, pcd2, rgb1_colors, rgb2_colors, corres1_3d, corres2_3d
        except Exception as e:
            logging.error(f"Error in estimate_relative_pose: {e}")
            return None, None, None, None, None, None, None

def load_image(path, is_depth=False, is_mask=False):
    try:
        path = os.path.abspath(path)
        if is_mask:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, -1 if is_depth else 1)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return img
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        return None

def main():
    data_root = DATA_ROOT
    rgb1_path = f"{data_root}/rgb/000000.png"
    depth1_path = f"{data_root}/depth/000000.png"
    mask1_path = f"{data_root}/masks/000000.png"
    rgb2_path = f"{data_root}/rgb/000011.png"
    depth2_path = f"{data_root}/depth/000011.png"
    mask2_path = f"{data_root}/masks/000011.png"
    cam_K_path = f"{data_root}/cam_K.txt"

    try:
        rgb1 = load_image(rgb1_path)
        depth1 = load_image(depth1_path, is_depth=True)
        mask1 = load_image(mask1_path, is_mask=True)
        rgb2 = load_image(rgb2_path)
        depth2 = load_image(depth2_path, is_depth=True)
        mask2 = load_image(mask2_path, is_mask=True)
        cam_K = np.loadtxt(cam_K_path).reshape(3, 3)
        
        if any(x is None for x in [rgb1, depth1, mask1, rgb2, depth2, mask2]):
            logging.error("One or more input images failed to load")
            return
    except Exception as e:
        logging.error(f"Error loading inputs: {e}")
        return

    try:
        tracker = PoseTracker()
    except Exception as e:
        logging.error(f"Failed to initialize PoseTracker: {e}")
        return

    relative_pose, pcd1, pcd2, rgb1_colors, rgb2_colors, corres1_3d, corres2_3d = tracker.estimate_relative_pose(
        rgb1, depth1, mask1, rgb2, depth2, mask2, cam_K
    )
    if relative_pose is not None:
        print("Relative pose (second camera w.r.t. first):\n", relative_pose)
        visualize_point_clouds(pcd1, pcd2, rgb1_colors, rgb2_colors, corres1_3d, corres2_3d, relative_pose)
    else:
        logging.error("Failed to estimate relative pose")

if __name__ == "__main__":
    main()