# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

import os
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from tqdm import tqdm

def collect_files(base_path, prefix):
    """Collect all files starting with the specified prefix."""
    return [f for f in os.listdir(base_path) if f.startswith(prefix)]


if __name__ == "__main__":
    
    # Path to the input image
    base_path = '../mm-test-detection-samples/'
    prefix = 'elevator'
    
    # Collect all files that start with 'water-fountain'
    files = collect_files(base_path, prefix)

    logging.info("Initialize object detectors")
    gdino = GroundingDINOObjectPredictor()
    SAM = SegmentAnythingPredictor()

    text_prompt = 'Water Cooler' # 'Floor selection button' # need to check for elevator button

    for file in tqdm(files):
        
        logging.info("Open the image and convert to RGB format")
        image_path = os.path.join(base_path, file)
        image_pil = PILImg.open(image_path).convert("RGB")
        
        logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        logging.info("GDINO post processing")
        w, h = image_pil.size # Get image width and height 
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

        # bbox_annotated_pil.show()
        out_file = os.path.join(f"{image_path}.pred.png")
        bbox_annotated_pil.save(out_file)
