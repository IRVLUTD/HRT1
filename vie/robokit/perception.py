#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import os
import torch
import hydra
import logging
import warnings
import numpy as np
from PIL import Image as PILImg
import torchvision.transforms as tvT
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import predict
from groundingdino.util.utils import clean_state_dict
# from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from matplotlib import (patches, pyplot as plt)
import matplotlib.cm as cm

# Depth Point Cloud
import math
import pyrender
from sklearn.neighbors import KDTree


os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")


class Logger(object):
    """
    This is a logger class.

    Attributes:
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Logger class.
        """
        super(Logger, self).__init__()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class Device(object):
    """
    This is a device class.

    Attributes:
        device (str): The device type ('cuda' or 'cpu').
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Device class.
        """
        super(Device, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class CommonContextObject(Logger, Device):
    """
    This is a common context object class.

    Attributes:
        logger: Logger instance for logging.
        device (str): The device type ('cuda' or 'cpu').
    """
    def __init__(self):
        """
        Initializes the CommonContextObject class.
        """
        super(CommonContextObject, self).__init__()


class ObjectPredictor(CommonContextObject):
    """
    Root class for object predicton
    All other object prediction classes should inherit this
    """
    def __init__(self):
        super(ObjectPredictor, self).__init__()
    

    def bbox_to_scaled_xyxy(self, bboxes: torch.tensor, img_w, img_h):
        """
        Convert bounding boxes to scaled xyxy format.

        Parameters:
        - bboxes (torch.tensor): Input bounding boxes in cxcywh format.
        - img_w (int): Image width.
        - img_h (int): Image height.

        Returns:
        - torch.tensor: Converted bounding boxes in xyxy format.
        """
        try:
            bboxes = bboxes * torch.Tensor([img_w, img_h, img_w, img_h])
            bboxes_xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
            return bboxes_xyxy
        
        except Exception as e:
            self.logger.error(f"Error during bounding box conversion: {e}")
            raise e


class GroundingDINOObjectPredictor(ObjectPredictor):
    """
    This class implements Object detection using HuggingFace GroundingDINO
    Here instead of using generic language query, we fix the text prompt as "objects" which enables
    getting compact bounding boxes arounds generic objects.
    Hope is that these cropped bboxes when used with OpenAI CLIP yields good classification results.
    """
    def __init__(self):
        super(GroundingDINOObjectPredictor, self).__init__()
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
        self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"
        self.model = self.load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filenmae
        )
    

    def load_model_hf(self, model_config_path, repo_id, filename):
        """
        Load model from Hugging Face hub.

        Parameters:
        - model_config_path (str): Path to model configuration file.
        - repo_id (str): ID of the repository.
        - filename (str): Name of the file.

        Returns:
        - torch.nn.Module: Loaded model.
        """
        try:
            args = SLConfig.fromfile(model_config_path) 
            model = build_model(args)
            args.device = self.device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location=self.device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model    

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise e

    def image_transform_grounding(self, image_pil):
        """
        Apply image transformation for grounding.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - tuple: Tuple containing original PIL image and transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image_pil, image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for grounding: {e}")
            raise e

    def image_transform_for_vis(self, image_pil):
        """
        Apply image transformation for visualization.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - torch.tensor: Transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for visualization: {e}")
            raise e
    
    def predict(self, image_pil: PILImg, det_text_prompt: str = "objects"):
        """
        Get predictions for a given image using GroundingDINO model.
        Paper: https://arxiv.org/abs/2303.05499
        Parameters:
        - image_pil (PIL.Image): PIL.Image representing the input image.
        - det_text_prompt (str): Text prompt for object detection
        Returns:
        - bboxes (list): List of normalized bounding boxeS in cxcywh
        - phrases (list): List of detected phrases.
        - conf (list): List of confidences.

        Raises:
        - Exception: If an error occurs during model prediction.
        """
        try:
            _, image_tensor = self.image_transform_grounding(image_pil)
            bboxes, conf, phrases = predict(self.model, image_tensor, det_text_prompt, box_threshold=0.3, text_threshold=0.35, device=self.device)
            return bboxes, phrases, conf        
        except Exception as e:
            self.logger.error(f"Error during model prediction: {e}")
            raise e


class SegmentAnythingPredictor(ObjectPredictor):
    """
    Predictor class for segmenting objects using the Segment Anything model.

    Inherits from ObjectPredictor.

    Attributes:
    - device (str): The device used for inference, either "cuda" or "cpu".
    - sam (torch.nn.Module): The Segment Anything model.
    - mask_generator (SamAutomaticMaskGenerator): The mask generator for the SAM model.
    - predictor (SamPredictor): The predictor for the SAM model.
    """

    def __init__(self):
        """
        Initialize the SegmentAnythingPredictor object.
        """
        super(SegmentAnythingPredictor, self).__init__()
        self.sam = sam_model_registry["vit_t"](checkpoint="ckpts/mobilesam/vit_t.pth")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)  # generate masks for entire image
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    def predict(self, image, prompt_bboxes):
        """
        Predict segmentation masks for the input image.

        Parameters:
        - image: The input image as a numpy array.
        - prompt_bboxes: Optional prompt bounding boxes as a list of lists of integers [x_min, y_min, x_max, y_max].

        Returns:
        - A tuple containing the input bounding boxes (if provided) and the segmentation masks as torch Tensors.

        Raises:
        - ValueError: If the input image is not a numpy array.
        """
        try:
            # Convert input image to numpy array
            image = np.array(image)

            # Check if prompt_bboxes is provided
            if prompt_bboxes is not None:
                # Convert prompt bounding boxes to torch tensor
                input_boxes = torch.tensor(prompt_bboxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                self.predictor.set_image(image)
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                input_boxes = None
                masks = self.mask_generator.generate(image)
            
            return input_boxes, masks

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None, None


class SAM2VideoPredictor(ObjectPredictor):
    """
    Predictor class for video object segmentation using the SAM2 model.
    Source: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/notebooks/video_predictor_example.ipynb
    """
    def __init__(self, text_prompt=None):
        """
        Initializes the SAM2VideoPredictor class and attempts to load the model.
        """
        self.logger = logging.getLogger(__name__)        
        self.predictor = self._load_predictor()
        self.text_prompt = text_prompt


    def init_hydra_and_model_setup(self):
        # Source: https://github.com/facebookresearch/sam2/issues/81#issuecomment-2262979343
        # hydra is initialized on import of sam2, which sets the search path which can't be modified
        # so we need to clear the hydra instance
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        # reinit hydra with a new search path for configs
        hydra.initialize_config_module("robokit/sam2/sam2/", version_base='1.2') # Please don't change this

        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # Please don't change this
        self.checkpoint_path = "./ckpts/samv2/sam2.1_hiera_large.pth" # Please don't change this


    def _load_predictor(self):
        """
        Load the SAM2 model using the configuration and checkpoint path.
        """
        try:
            self.init_hydra_and_model_setup()

            # Load the SAM2 model with the configuration and checkpoint
            predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path)

            print("SAM2 video predictor initialized successfully.")
            
            return predictor
        
        except Exception as e:
            print(f"Failed to load the predictor: {e}")
            raise

    def _load_and_show_frame(self, video_dir, frame_idx=0, show_frame=False):
        """
        Load and optionally display a specific video frame from a directory containing JPEG frames.

        Parameters:
        - video_dir: Directory containing video frames.
        - frame_idx: Index of the frame to display (default is 0).
        - show_frame: Boolean flag to indicate whether to display the frame (default is False).

        Returns:
        - img: The loaded image.
        """
        # Scan all image frames in the directory
        self.frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in self.get_valid_img_extensions()
        ]
        
        # Sort frames by the numeric part of the filename
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Load the specified frame
        if frame_idx < len(self.frame_names):
            frame_path = os.path.join(video_dir, self.frame_names[frame_idx])
            img = PILImg.open(frame_path)

            # Display the frame if requested
            if show_frame:
                plt.figure(figsize=(9, 6))
                plt.title(f"Frame {frame_idx}")
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            return img
        else:
            print(f"Frame index {frame_idx} is out of range.")
            return None
    

    def rename_png_to_jpg(video_dir):
        """
        Rename all PNG images in the specified video directory to JPG,
        replacing '_color' with an empty string in the filenames.
        This is required as SAMv2 only reads mp4 and jpg/jpeg/JPG/JPEG files with frame names as 00000.jpg-....
        Parameters:
        - video_dir: Directory containing PNG images.
        """
        for filename in os.listdir(video_dir):
            if filename.endswith(".png"):
                # Replace '_color' with '' and change the file extension to .jpg
                new_filename = filename.replace('_color', '').replace('.png', '.jpg')
                old_path = os.path.join(video_dir, filename)
                new_path = os.path.join(video_dir, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")


    def segment_using_bbox(self, video_dir, frame_idx, bbox):
        """
        Segment an object from the video frame using a bounding box.
        
        Parameters:
        - video_dir: Path to video frames directory
        - frame_idx: Index of the frame to work with
        - bbox: The bounding box [x_min, y_min, x_max, y_max]
        """
        try:
            """
            SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
            During initialization, it loads all the JPEG frames in video_path and stores their pixels in inference_state
            """

            inference_state = self.predictor.init_state(video_path=video_dir)

            # Note: if you have run any previous tracking using this inference_state, please reset it first via reset_state.
            self.predictor.reset_state(inference_state)

            # Load the video frames
            self._load_and_show_frame(video_dir, show_frame=True)

            # Load the frame
            frame_names = self.load_frames_from_directory(video_dir)
            frame_path = os.path.join(video_dir, frame_names[frame_idx])
            image = PILImg.open(frame_path)
            plt.imshow(image)
            plt.title(f"Frame {frame_idx}")

            # Perform segmentation using the bounding box
            ann_obj_id = 1  # Unique identifier for the object
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                box=bbox
            )
            self.show_box(bbox, plt.gca())
            self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            plt.show()

        except Exception as e:
            print(f"Error during segmentation: {e}")
            raise


    def save_binary_mask_with_pil(self, mask, save_path):
        """
        Saves a binary mask as a PNG file using PIL while preserving binary values (0 and 1).

        Args:
            mask (np.ndarray): A 2D NumPy array representing the binary mask (values 0 and 1).
            save_path (str): The file path where the PNG image will be saved.
        """
        try:
            # Ensure the mask is in uint8 format
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            mask[mask > 0] = 255

            # Convert the NumPy array to a PIL image
            mask_image = PILImg.fromarray(mask)

            # Save the mask as a PNG file
            mask_image.save(save_path, format="PNG")
            print(f"Binary mask saved successfully at {save_path}")
        except Exception as e:
            print(f"Error saving binary mask: {e}")

    def propagate_masks_and_save(self, video_dir, bbox, vis_frame_stride=1, save_output=True):
        """
        Propagate the segmentation mask across the entire video and optionally save the frames with masks to a subdirectory.
        
        Parameters:
        - video_dir: Path to the video frames directory.
        - bbox: The bounding box [x_min, y_min, x_max, y_max] for initial segmentation.
        - vis_frame_stride: Number of frames to skip while visualizing the segmentation (default is 30).
        - save_output: If True, saves the segmented frames (default is True).
        """
        try:
            # Initialize inference state
            inference_state = self.predictor.init_state(video_path=video_dir)
            self.predictor.reset_state(inference_state)
            
            # Get all frames from the directory
            frame_names = self.load_frames_from_directory(video_dir)
            
            # Segment first frame
            frame_idx = 0
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                box=bbox
            )
            
            # Create output directory if save_output is True
            if save_output:
                out_path_suffix = f"/{self.text_prompt.lower().replace(' ', '_')}" if self.text_prompt else ''
                output_dir = os.path.join(os.path.dirname(video_dir), f"out/samv2{out_path_suffix}")
                masks_dir = os.path.join(output_dir, "obj_masks")
                traj_overlayed_dir = os.path.join(output_dir, "masks_traj_overlayed")
                os.makedirs(masks_dir, exist_ok=True)
                os.makedirs(traj_overlayed_dir, exist_ok=True)

            # Initialize trajectory list
            centroids = []
            
            # Propagate mask across video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()}
                    # logging.info("Visualize and optionally save the results")
                    plt.close("all")
            
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Frame {out_frame_idx}")
                    img_path = os.path.join(video_dir, frame_names[out_frame_idx])
                    img = PILImg.open(img_path)
                    plt.imshow(img)
                    
                    # Draw bounding box
                    plt.gca().add_patch(plt.Rectangle(
                        (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                        linewidth=2, edgecolor="cyan", facecolor="none"))
                    
                    out_file_name = frame_names[out_frame_idx].replace('.jpg', '.png')
                    
                    # Show segmentation masks
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                        
                        # assumption only one object exists
                        self.save_binary_mask_with_pil(out_mask[0], os.path.join(masks_dir, out_file_name))

                        centroid =  self.calculate_centroid(out_mask)
                        centroids.append(centroid)

                        # Plot the tracklet with centroids
                        num_centroids = len(centroids)
                        colormap = cm.get_cmap("autumn")  # Choose a colormap
                        
                        for idx, (x, y) in enumerate(centroids):
                            color = colormap(idx / num_centroids)  # Darker for more recent, lighter for older
                            plt.plot(x, y, 'o', color=color, markersize=3)
                        
                        # Disable axis numbers and ticks
                        plt.axis('off')
                        
                        # Save the trajectory overlayed image if save_output is True
                        if save_output:
                            traj_result_path = os.path.join(traj_overlayed_dir, out_file_name)
                            plt.savefig(traj_result_path)
                            plt.close()
            
        except Exception as e:
            logging.error(f"Error during mask propagation: {e}")

        return frame_names, video_segments

    def calculate_centroid(self, mask):
        """
        Calculate the centroid of the object in the mask.
        """
        # Assuming the mask is a binary numpy array with the object pixels as 1
        _, y_indices, x_indices = np.where(mask == 1)
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        return (centroid_x, centroid_y)



    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        """
        Displays the segmentation mask on the given axes.
        
        Parameters:
        - mask: The segmentation mask.
        - ax: The axes on which to display the mask.
        - obj_id: Optional object ID to color code the mask.
        - random_color: If True, assigns a random color to the mask.
        """
        try:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                color = np.array([*cmap(obj_id or 0)[:3], 0.6])

            mask_image = mask.reshape(mask.shape[-2], mask.shape[-1], 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        except ValueError as e:
            self.logger.error(f"Error in show_mask: {e}")
            raise

    def show_points(self, coords, labels, ax, marker_size=200):
        """
        Displays points on the image based on their labels.
        
        Parameters:
        - coords: Coordinates of the points.
        - labels: Labels for the points (e.g., positive or negative).
        - ax: The axes for displaying the points.
        - marker_size: The size of the point markers.
        """
        try:
            pos_points = coords[labels == 1]
            neg_points = coords[labels == 0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        except IndexError as e:
            self.logger.error(f"Error in show_points: {e}")
            raise

    def show_box(self, box, ax):
        """
        Draws a bounding box on the axes.
        
        Parameters:
        - box: Bounding box in the format [x_min, y_min, x_max, y_max].
        - ax: The axes to draw the bounding box on.
        """
        try:
            if len(box) != 4:
                raise ValueError("Box must contain exactly 4 values: [x_min, y_min, x_max, y_max].")
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        except ValueError as e:
            self.logger.error(f"Error in show_box: {e}")
            raise

    def get_valid_img_extensions(self):
        """
        Returns a list of valid image file extensions.
        """
        return [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def load_frames_from_directory(self, video_dir):
        """
        Loads frame names from the directory that are valid image files.

        Parameters:
        - video_dir: Directory containing video frames.

        Returns:
        - List of valid image file names.
        """
        valid_extensions = self.get_valid_img_extensions()
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in valid_extensions
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names

    def create_collage(self, video_dir, collage_size=(2, 3)):
        """
        Creates a collage of video frames for visualization.
        
        Parameters:
        - video_dir: Directory of video frames.
        - collage_size: The number of rows and columns in the collage.
        """
        try:
            frame_names = self.load_frames_from_directory(video_dir)
            fig, axes = plt.subplots(collage_size[0], collage_size[1], figsize=(12, 8))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < len(frame_names):
                    img = PILImg.open(os.path.join(video_dir, frame_names[i]))
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error creating collage: {e}")
            raise

class DepthPointCloud:
    """
    Source: https://github.com/IRVLUTD/hamer-depth/blob/070886168e469ab1645612a2c3b8c6473aab1aef/mesh_to_sdf/depth_point_cloud.py
    Credits: Yu Xiang
    """
    def __init__(self, depth, intrinsic_matrix, camera_pose, target_mask=None, threshold=1.5):
        self.depth = depth
        self.intrinsic_matrix = intrinsic_matrix
        self.camera_pose = camera_pose
        self.target_mask = target_mask
        self.width = depth.shape[1]
        self.height = depth.shape[0]
        self.threshold = threshold

        # backproject to camera
        pc = self.backproject_camera(depth, intrinsic_matrix)

        # transform points to world
        pc_base = camera_pose[:3, :3] @ pc + camera_pose[:3, 3].reshape((3, 1))
        self.points = pc_base.T
        self.kd_tree = KDTree(self.points)
        

    def get_random_surface_points(self, count):
        indices = np.random.choice(self.points.shape[0], count)
        return self.points[indices, :]


    def backproject_camera(self, im_depth, K):  
        Kinv = np.linalg.inv(K)

        width = im_depth.shape[1]
        height = im_depth.shape[0]
        depth = im_depth.astype(np.float32, copy=True).flatten()
        if self.target_mask is not None:
            mask = (depth > 0) & (depth < self.threshold) & (self.target_mask.flatten() == 0)
        else:
            mask = (depth > 0) & (depth < self.threshold)

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

        # backprojection
        R = Kinv.dot(x2d.transpose())
        X = np.multiply(
            np.tile(depth.reshape(1, width * height), (3, 1)), R
        )
        return X[:, mask]
    

    # query points are in world frame
    def get_sdf(self, query_points):
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1)
        inside = ~self.is_outside(query_points)
        distances[inside] *= -1
        return distances
    
    
    def get_sdf_cost(self, query_points, epsilon=0.02, w_inside=1, vis=False):
        print('computing sdf cost...')
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1)
        inside = ~self.is_outside(query_points)
        distances[inside] *= -1

        # visualization
        if vis:
            index = np.absolute(distances) < 0.03
            points_show = query_points[index]
            colors = np.zeros(points_show.shape)
            colors[distances[index] < 0, 2] = 1
            colors[distances[index] > 0, 0] = 1
            cloud = pyrender.Mesh.from_points(points_show, colors=colors)
            scene = pyrender.Scene()
            scene.add(cloud)
            scene.add(pyrender.Mesh.from_points(self.points[::100]))
            pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

        # cost
        cost = np.zeros_like(distances)
        cost[inside] = w_inside * (-distances[inside] + epsilon / 2)
        index = (distances > 0) & (distances < epsilon)
        cost[index] = np.square(distances[index] - epsilon) / (2 * epsilon)
        print('done')
        return cost


    def get_sdf_in_batches(self, query_points, batch_size=1000000):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points)
            for points in np.array_split(query_points, n_batches)
        ]
        return np.concatenate(batches)


    def show(self):
        # compute sdf for sampled points
        query_points = []
        surface_sample_count = 10000
        surface_points = self.get_random_surface_points(surface_sample_count)
        query_points.append(surface_points + np.random.normal(scale=0.025, size=(surface_sample_count, 3)))
        query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))        
        query_points = np.concatenate(query_points).astype(np.float32)
        sdf = self.get_sdf(query_points)

        # visualization
        colors = np.zeros(query_points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        cloud = pyrender.Mesh.from_points(query_points, colors=colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        scene.add(pyrender.Mesh.from_points(self.points))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


    def is_outside(self, points):
        # project points to camera view
        RT = np.linalg.inv(self.camera_pose)
        pc = points.T
        pc_camera = RT[:3, :3] @ pc + RT[:3, 3].reshape((3, 1))
        x2d = self.intrinsic_matrix @ pc_camera
        x2d[0, :] /= x2d[2, :]
        x2d[1, :] /= x2d[2, :]
        pixels = x2d[:2].T.astype(int)

        # This only has an effect if the camera is inside the model
        in_viewport = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < self.width) & (pixels[:, 1] < self.height)
        pc_camera = pc_camera.T
        result = np.ones(points.shape[0], dtype=bool)
        result[in_viewport] = pc_camera[in_viewport, 2] < self.depth[pixels[in_viewport, 1], pixels[in_viewport, 0]]
        return result
