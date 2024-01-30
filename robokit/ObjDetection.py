import os
import clip
import torch
import logging
import warnings
from PIL import Image as PILImg
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict

os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")


class Logger:
    """
    This is a logger class
    """
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    

class ObjectDetector(Logger):
    """
    Root class for object detection
    All other object detector classes should inherit this
    """
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class GroundingDINOObjectDetector(ObjectDetector):
    """
    This class implements Object detection using HugginFace GroundingDINO
    Here instead of using generic language query, we fix the text prompt as "objects" which enables
    getting compact bounding boxes arounds generic objects.
    These cropped bboxes when used with OpenAI CLIP yields good classification results.
    """

    def __init__(self):
        super().__init__()
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
        self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"
        self.model = self.load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filenmae
        )


    def load_model_hf(self, model_config_path, repo_id, filename):
        args = SLConfig.fromfile(model_config_path) 
        model = build_model(args)
        args.device = self.device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model    

    def image_transform_grounding(self, image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image_pil, None) # 3, h, w
        return image_pil, image

    def image_transform_for_vis(self, image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        image, _ = transform(image_pil, None) # 3, h, w
        return image

    def bbox_to_scaled_xyxy(self, bboxes: torch.tensor, img_w, img_h):
        bboxes = bboxes * torch.Tensor([img_w, img_h, img_w, img_h])
        bboxes_xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        return bboxes_xyxy
    
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
            bboxes, conf, phrases = predict(self.model, image_tensor, det_text_prompt, box_threshold=0.25, text_threshold=0.25, device=self.device)
            return bboxes, phrases, conf        
        except Exception as e:
            self.logger.error(f"Error during model prediction: {e}")
            raise e


class ZeroShotClipPredictor(Logger):
    def __init__(self):
        super().__init__()

        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-L/14@336px', self.device)

    def get_features(self, images, text_prompts):
        """
        Extract features from a list of images and text prompts.

        Parameters:
        - images (list of PIL.Image): A list of PIL.Image representing images.
        - text_prompts (list of str): List of text prompts.

        Returns:
        - Tuple of numpy.ndarray: Concatenated image features and text features as numpy arrays.

        Raises:
        - ValueError: If images is not a tensor or a list of tensors.
        - RuntimeError: If an error occurs during feature extraction.
        """
        try:

            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)
                _images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                img_features = self.model.encode_image(_images)
                text_features = self.model.encode_text(text_inputs)
            
            return img_features, text_features

        except ValueError as ve:
            self.logger.error(f"ValueError in get_image_features: {ve}")
            raise ve
        except RuntimeError as re:
            self.logger.error(f"RuntimeError in get_image_features: {re}")
            raise re

    def predict(self, image_array, text_prompts):
        """
        Run zero-shot prediction using CLIP model.

        Parameters:
        - image_array (List[torch.tensor]): List of tensor images.
        - text_prompts (list): List of text prompts for prediction.

        Returns:
        None
        """
        try:

            image_features, text_features = self.get_features(image_array, text_prompts)


            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pconf, indices = similarity.topk(1)

            return (pconf.flatten(), indices.flatten())

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise e
