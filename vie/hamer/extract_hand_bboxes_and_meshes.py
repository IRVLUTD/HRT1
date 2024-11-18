#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
This script is modified from https://github.com/geopavlakos/hamer/blob/df533a2d04b9e2ece7cf9d6cbc6982e140210517/demo.py
"""

import os
import cv2
import torch
import argparse
import numpy as np
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


class HandMeshBoundingBoxExtractor:
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

    def extract_bounding_boxes(self, img_path: str, save_mesh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Get the root directory name (parent folder name)
        parent_dir = os.path.dirname(img_path)
        root_dir_name = os.path.basename(parent_dir)
        
        # Create the output folder with 'hamer/root_dir_name' suffix
        out_folder = os.path.normpath(os.path.join(parent_dir, f"../out/hamer/"))
        os.makedirs(out_folder, exist_ok=True)

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
        
        from PIL import Image, ImageDraw
        
        # Process pose outputs
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            # image = Image.fromarray(img_cv2[:,:,::-1])
            # draw = ImageDraw.Draw(image)

            # points = left_hand_keyp[:,:-1]

            # # Plot each point on the image
            # is_first = True
            # for x, y in points:
            #     # Convert coordinates to integers
            #     x, y = int(x), int(y)
            #     # Draw a small circle at each point
            #     radius = 3
            #     if is_first:
            #         a,b = "cyan", "magenta"
            #         is_first = False
            #     else:
            #         b,a = "magenta", "cyan"
            #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=a, outline=b)

            # points = right_hand_keyp[:,:-1]

            # # Plot each point on the image
            # is_first = True
            # for x, y in points:
            #     # Convert coordinates to integers
            #     x, y = int(x), int(y)
            #     # Draw a small circle at each point
            #     radius = 3
            #     if is_first:
            #         a,b = "red", "yellow"
            #         is_first = False
            #     else:
            #         b,a = "red", "yellow"
            #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=a, outline=b)

            # # Show the image with points overlayed
            # image.show()
            
            

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
            self._save_meshes(img_cv2, boxes, right, img_path, out_folder)

        return boxes, right

    def _save_meshes(self, img_cv2, boxes, right, img_path, out_folder):
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        import matplotlib.pyplot as plt

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            import pdb; pdb.set_trace()
            # # Convert tensor to numpy for easier handling
            # import numpy as np
            # from PIL import Image, ImageDraw
            # hand_pose_tensor = np.array(out['pred_keypoints_2d'].cpu())

            # # Create a drawing object to draw on the image
            # image = Image.fromarray(img_cv2)
            # draw = ImageDraw.Draw(image)

            # # Scale factor to convert normalized keypoints into image coordinates
            # scale_x = image.width
            # scale_y = image.height

            # # Plot the keypoints for each hand
            # colors = ['blue', 'red']  # Blue for hand 1, red for hand 2

            # for i in range(2):  # Iterate over the two hands
            #     hand_keypoints = hand_pose_tensor[i]
                
            #     # Plot the keypoints
            #     for x, y in hand_keypoints:
            #         # Convert normalized (x, y) to image coordinates
            #         img_x = int(x * scale_x)
            #         img_y = int(y * scale_y)
                    
            #         # Draw a small circle at the keypoint location
            #         radius = 3
            #         draw.ellipse((img_x - radius, img_y - radius, img_x + radius, img_y + radius), fill=colors[i], outline=colors[i])
                
            #     # Optionally, connect the keypoints to form a skeleton (lines between joints)
            #     for j in range(20):  # 20 connections for 21 keypoints
            #         x1, y1 = hand_keypoints[j]
            #         x2, y2 = hand_keypoints[j + 1]
                    
            #         # Convert to image coordinates
            #         img_x1, img_y1 = int(x1 * scale_x), int(y1 * scale_y)
            #         img_x2, img_y2 = int(x2 * scale_x), int(y2 * scale_y)
                    
            #         # Draw the line connecting the keypoints
            #         draw.line((img_x1, img_y1, img_x2, img_y2), fill=colors[i], width=2)

            # # Show the image with the keypoints drawn
            # image.show()
            # import pdb; pdb.set_trace()

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
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
                cv2.imwrite(os.path.join(out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])
                # """

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                # exit()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                camera_translation = cam_t.copy()
                tmesh = self.renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                out_path = os.path.join(out_folder, f'{img_fn}_{person_id}.obj')
                tmesh.export(out_path)
                print(f"Mesh saved to {out_path}")
            

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

            cv2.imwrite(os.path.join(out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])


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
            extractor = HandMeshBoundingBoxExtractor()

            # Process each image file
            for img_file in tqdm(image_files):
                img_path = os.path.join(input_dir, img_file)
                boxes, right = extractor.extract_bounding_boxes(img_path, save_mesh=True)
                print("Boxes:", boxes)
                print("Right:", right)  # 1 for right hand, 2 for left hand

