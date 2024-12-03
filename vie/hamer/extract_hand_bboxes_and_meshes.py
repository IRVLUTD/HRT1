#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
This script is modified from https://github.com/geopavlakos/hamer/blob/df533a2d04b9e2ece7cf9d6cbc6982e140210517/demo.py
The script in its default state is capable of providing hand mesh for multiple persons in the image.
We have not modified this ability. In our setup, we make sure that one scene contains only one person
"""

import os
import cv2
import torch
import argparse
import numpy as np


from scipy.optimize import minimize
import open3d as o3d
from mesh_to_sdf.rgbd2pc import RGBD2PC
import matplotlib.pyplot as plt


from pathlib import Path
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
from typing import Tuple
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from tqdm import tqdm

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)



import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_depth_img(img_path):
    """
    Loads a depth image corresponding to the given RGB image path.

    This function replaces the 'rgb' directory with 'depth' and the file 
    extension from '.jpg' to '.png' to locate the depth image. It reads the 
    depth image, normalizes it by dividing by 1000 (to convert the depth 
    values from millimeters to meters), and returns the depth data as a 
    NumPy array.

    Source: https://github.com/IRVLUTD/hamer-depth/commit/070886168e469ab1645612a2c3b8c6473aab1aef#diff-6bacd8700314864adb2bf1d56bb841dab8e0ac87d88c8303caa83b545d0b4b9dR116

    Args:
        img_path (str): Path to the RGB image file.

    Returns:
        np.ndarray: Normalized depth image as a NumPy array.

    Raises:
        FileNotFoundError: If the depth image file does not exist.
        ValueError: If the depth image cannot be loaded or is invalid.
    """
    try:
        # Replace 'rgb' with 'depth' and change the extension to '.png'
        depth_path = str(img_path).replace('rgb', 'depth').replace('jpg', 'png')
        logger.info(f"Attempting to load depth image from: {depth_path}")

        # Read the depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to load depth image from {depth_path}")

        # Convert depth to float32 and normalize
        depth = depth.astype(np.float32) / 1000.0
        logger.info("Depth image loaded and normalized successfully.")

        return depth

    except FileNotFoundError as e:
        logger.error(f"Depth image file not found: {e}")
        raise

    except ValueError as e:
        logger.error(f"Error loading depth image: {e}")
        raise

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the depth image: {e}")
        raise



def get_my_intrinsic_matrix():
    """
    Retrieves the intrinsic matrix of the target camera.

    The intrinsic matrix describes the camera's internal parameters, including
    focal lengths and the principal point. This function returns the intrinsic 
    matrix of the IRVL Fetch robot camera.

    Returns:
        np.ndarray: A 3x3 intrinsic matrix of the target camera.

    Example:
        >>> intrinsic_matrix = get_my_intrinsic_matrix()
        >>> print(intrinsic_matrix)
        [[574.0528    0.      319.5   ]
         [  0.      574.0528  239.5   ]
         [  0.        0.        1.    ]]
    """
    logger.info("Retrieving the intrinsic matrix of the IRVL Fetch robot camera.")
    intrinsic_matrix = np.array([
        [574.0527954101562, 0.0, 319.5],
        [0.0, 574.0527954101562, 239.5],
        [0.0, 0.0, 1.0]
    ])
    logger.info("Intrinsic matrix successfully retrieved.")
    return intrinsic_matrix



def obj_function(x, vertices, translation, K1, K2, kd_tree, weight_3d=100):
    """
    Computes the objective function value for optimization.

    This function calculates the combined error, which includes:
    - 2D projection error between two sets of 3D vertices projected onto 
      camera planes using intrinsic matrices.
    - Weighted 3D distance error between the translated vertices and 
      their nearest neighbors in a KD-tree.

    Args:
        x (np.ndarray): A 3D translation vector for the second projection (shape: (3,)).
        vertices (np.ndarray): Nx3 array of 3D points (vertices).
        translation (np.ndarray): A 3D translation vector for the first projection (shape: (3,)).
        K1 (np.ndarray): 3x3 intrinsic matrix for the first camera.
        K2 (np.ndarray): 3x3 intrinsic matrix for the second camera.
        kd_tree (scipy.spatial.KDTree): KD-tree for nearest-neighbor search.
        weight_3d (float, optional): Weight factor for the 3D error. Defaults to 100.

    Returns:
        float: The combined error value (2D projection error + weighted 3D error).
    """
    try:
        # Projection 1
        V1 = vertices + translation
        x1 = K1 @ V1.T
        x1[0, :] /= x1[2, :]
        x1[1, :] /= x1[2, :]
        logger.debug("Projection 1 completed.")

        # Projection 2
        V2 = vertices + x
        x2 = K2 @ V2.T
        x2[0, :] /= x2[2, :]
        x2[1, :] /= x2[2, :]
        logger.debug("Projection 2 completed.")

        # Compute 3D distances
        distances, _ = kd_tree.query(V2)
        distances = distances.astype(np.float32).reshape(-1)
        error_3d = np.mean(distances)
        logger.debug(f"3D distance error: {error_3d}")

        # Compute 2D projection error
        error_2d = np.square(x1[:2] - x2[:2]).mean()
        logger.debug(f"2D projection error: {error_2d}")

        # Combine errors with weighting
        total_error = error_2d + weight_3d * error_3d
        logger.info(f"Total error: {total_error} (2D: {error_2d}, 3D: {error_3d}, Weight: {weight_3d})")

        return total_error

    except Exception as e:
        logger.error(f"An error occurred while computing the objective function: {e}")
        raise


class HandInfoExtractor:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, body_detector: str = 'vitdet', rescale_factor: float = 2.0):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model = self.model.to(self.device).eval()
        self.detector = self._initialize_detector(body_detector)
        self.cpm = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.rescale_factor = rescale_factor

    def _initialize_detector(self, body_detector: str):
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

    def extract_info(self, img_path: str, save_mesh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Get the root directory name (parent folder name)
        parent_dir = os.path.dirname(img_path)
        
        # Read the image
        img_cv2 = cv2.imread(img_path)
        
        # Perform detection (assuming `self.detector` is initialized elsewhere)
        det_out = self.detector(img_cv2)
        img = img_cv2[:, :, ::-1].copy()

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Predict poses (assuming `self.cpm` is initialized elsewhere)
        vitposes_out = self.cpm.predict_pose(
            img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        # Initialize bounding box list and right/left hand flags
        bboxes = []
        is_right = []
        
        # Process pose outputs
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            for keyp, right_flag in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:  # Ensure enough valid keypoints
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(right_flag)

        # If no bounding boxes found, return empty arrays
        if not bboxes:
            return np.array([]), np.array([])

        # Stack the bounding boxes and right/left hand flags
        boxes = np.stack(bboxes)
        right = np.stack(is_right)  # 1 for right hand, 0 for left hand

        # Optionally save the meshes
        if save_mesh:
            output_path, pcd, out = self._save_meshes(img_cv2, boxes, right, img_path, parent_dir)
        else:
            output_path, pcd, out = None, None, None
        
        return boxes, right, output_path, pcd, out

    def convert_output_to_numpy(self, out):
        """
        Convert the output dictionary with PyTorch tensors and additional data (bboxes, is_right) to NumPy arrays.
        
        Args:
            out (dict): The output dictionary containing model outputs, including 'pred_mano_params'.
        
        Returns:
            dict: The dictionary with all tensors and additional data converted to NumPy arrays.
        """
        # Convert tensors in the main output dictionary to NumPy arrays
        out_numpy = {k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in out.items()}
        
        # Specifically handle 'pred_mano_params' if it exists
        if 'pred_mano_params' in out_numpy:
            out_numpy['pred_mano_params'] = {
                k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in out_numpy['pred_mano_params'].items()
            }
        
        return out_numpy


    def save_point_cloud_as_ply(self, vertices, output_folder, filename, colors=None):
        """
        Save a point cloud (with optional RGB colors) as a PLY file using Open3D.

        Args:
            vertices (numpy.ndarray): Nx3 array of 3D points.
            output_folder (str): Directory to save the PLY file.
            filename (str): Name of the output PLY file (without extension).
            colors (numpy.ndarray, optional): Nx3 array of RGB colors (values in range [0, 255]).
                                            If None, saves the point cloud without colors.

        Returns:
            str: Full path to the saved PLY file.
            open3d.geometry.PointCloud: The Open3D PointCloud object created and saved.
        
        Raises:
            ValueError: If the number of color entries does not match the number of vertices.
        """
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # Add colors if provided
        if colors is not None:
            if colors.shape[0] != vertices.shape[0]:
                raise ValueError("The number of color entries must match the number of vertices.")
            # Normalize RGB values to [0, 1] range
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Construct the full file path for the PLY file
        output_path = os.path.join(output_folder, f"{filename}.ply")

        # Save the point cloud to a PLY file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Point cloud saved to '{output_path}' using Open3D.")

        return output_path, pcd


    def save_output_as_npz(self, out, bboxes, is_right, filepath):
        """
        Save the model output dictionary, bounding boxes, and hand flags to an .npz file after converting to NumPy arrays.
        
        Args:
            out (dict): The model output dictionary.
            bboxes (list or np.ndarray): The bounding boxes.
            is_right (list or np.ndarray): The right/left hand flags (1 for right hand, 0 for left hand).
            filepath (str): Path to the .npz file where the output should be saved.
        """
        # Convert all tensors and additional data (bboxes, right) to NumPy arrays
        out_numpy = self.convert_output_to_numpy(out)

        # Add bounding boxes and right/left hand flags to the output dictionary
        out_numpy['bboxes'] = np.stack(bboxes)  # Stack the bounding boxes into a NumPy array
        out_numpy['right'] = np.stack(is_right)  # Stack the right/left hand flags (1 for right hand, 0 for left hand)

        # Save the dictionary as an .npz file
        np.savez_compressed(filepath, **out_numpy)
        print(f"Output saved to {filepath}")

    def _save_meshes(self, img_cv2, boxes, right, img_path, parent_dir):
        # Create the output folder with 'hamer/root_dir_name' suffix
        out_root_dir = "../out/hamer"
        plots_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/extra_plots")) # for plots and objs
        model_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/model")) # for model output
        _3dhand_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/3dhand")) # for hand aligned with fetch cam
        scene_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/scene")) # scene point cloud
        os.makedirs(plots_out_folder, exist_ok=True)
        os.makedirs(model_out_folder, exist_ok=True)
        os.makedirs(_3dhand_out_folder, exist_ok=True)
        os.makedirs(scene_out_folder, exist_ok=True)

        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        depth = load_depth_img(img_path)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            
            # self.model_cfg.EXTRA.FOCAL_LENGTH = 574
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()
                
                # (easteregg) uncomment to plot hand cropped images
                # """
                regression_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                )
            
                side_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                            out['pred_cam_t'][n].detach().cpu().numpy(),
                            white_img,
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            side_view=True)
                
                final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)

                # final_img = np.concatenate([input_patch, regression_img], axis=1)

                # save image with mesh overlayed
                cv2.imwrite(os.path.join(plots_out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])
                # """

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(plots_out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

            # get the hand points
            mask = cam_view[:,:,3] > 0
            # eroder mask
            kernel = np.ones((5, 5), np.uint8) 
            mask = cv2.erode(mask.astype(np.uint8), kernel)
            mask = 1 - mask 
            intrinsic_matrix = get_my_intrinsic_matrix()

            # convert depth to point cloud
            depth_pc = RGBD2PC(depth, intrinsic_matrix, camera_pose=np.eye(4), target_mask=mask, threshold=10.0)

            # solve new translation
            scaled_focal_length = scaled_focal_length.item()
            K = np.array([[scaled_focal_length, 0, 320], [0, scaled_focal_length, 240], [0, 0, 1]]).astype(np.float32)
            x0 = np.mean(depth_pc.points, axis=0)
            
            res = minimize(obj_function, x0, method='nelder-mead',
                        args=(all_verts[-1], all_cam_t[-1], K, intrinsic_matrix, depth_pc.kd_tree), options={'xatol': 1e-8, 'disp': True})
            translation_new = res.x

            out['opt_translation'] = translation_new

            # save the model output
            self.save_output_as_npz(out, boxes, right, f"{model_out_folder}/{img_fn}.npz")

            # fig = plt.figure()
            # ax = fig.add_subplot(1, 3, 1)
            # plt.imshow(input_img)
            
            # # verify projection 1
            # vertices = all_verts[-1] + all_cam_t[-1]
            # print(K, vertices)
            # print(vertices.shape)
            # x2d = K @ vertices.T
            # x2d[0, :] /= x2d[2, :]
            # x2d[1, :] /= x2d[2, :]
            # plt.plot(x2d[0, :], x2d[1, :])
            # plt.title('projection using hamer camera')

            # ax = fig.add_subplot(1, 3, 2)
            # plt.imshow(input_img)

            # verify projection 2
            # vertices = all_verts[-1] + translation_new
            # x2d = intrinsic_matrix @ vertices.T
            # x2d[0, :] /= x2d[2, :]
            # x2d[1, :] /= x2d[2, :]
            # plt.plot(x2d[0, :], x2d[1, :])              
            # plt.title('projection using fetch camera')

            # ax = fig.add_subplot(1, 3, 3, projection='3d')
        
            # ax.scatter(depth_pc.points[:, 0], depth_pc.points[:, 1], depth_pc.points[:, 2], marker='o')
            # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', color='r')

            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()               

            # save rgbd scene pc
            RT_file = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'pose', f'{img_fn}.npz')
            RT = np.load(RT_file)['RT_camera'] # not using as of now
            img_cv2 = img_cv2.astype(np.float32)[:, :, ::-1]  # Convert from BGR to RGB
            RT = np.eye(4)
            scene_pcd = RGBD2PC(depth, intrinsic_matrix, rgb=img_cv2, camera_pose=RT, target_mask=None, threshold=10.0)
            scene_pcd.save_point_cloud(os.path.join(scene_out_folder, f"{img_fn}.ply"))

            # save fetch cam aligned hamer hand mesh pc
            output_path, pcd = self.save_point_cloud_as_ply(all_verts[-1] + translation_new, _3dhand_out_folder, f"{img_fn}")

        return output_path, pcd, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for hand mesh bounding box extraction.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images.")
    args = parser.parse_args()

    input_dir = args.input_dir

    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(f"Error: The directory '{input_dir}' does not exist or is not a valid directory.")
    else:
        # List all files in the directory and filter for image files (jpg, jpeg, png)
        image_files = [f for f in os.listdir(input_dir)
                       if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png'))]

        if not image_files:
            raise Exception(f"No image files found in the directory '{input_dir}'.")
        else:
            extractor = HandInfoExtractor()

            # Process each image file
            for img_file in tqdm(image_files):
                img_path = os.path.join(input_dir, img_file)
                import pdb; pdb.set_trace()
                boxes, right, output_path, pcd, out = extractor.extract_info(img_path, save_mesh=True)
