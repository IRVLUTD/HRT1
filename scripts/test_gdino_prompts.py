#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import os
import numpy as np
from absl import app, flags, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor
from tqdm import tqdm

# Set up absl flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None, 'Directory path to input images')
flags.DEFINE_string('text_prompt', None, 'Text prompt for GDINO predictions')

def main(argv):
    # Get the input directory and text prompt from FLAGS
    _image_root_dir = FLAGS.input_dir
    text_prompt = FLAGS.text_prompt

    if not _image_root_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()

        # Set output directory in the parent directory of _image_root_dir
        parent_dir = os.path.dirname(_image_root_dir)
        out_path_suffix = f"gdino/{text_prompt.lower().replace(' ', '_')}"
        out_path = os.path.join(parent_dir, out_path_suffix)
        os.makedirs(out_path, exist_ok=True)

        # Dummy mask for annotate func later on (we are using only GDINO and not SAM)
        dummy_masks = np.array([])

        img_files = os.listdir(_image_root_dir)

        for img_file in tqdm(img_files):
            image_path = os.path.join(_image_root_dir, img_file)

            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")

            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

            logging.info("GDINO post processing")
            w, h = image_pil.size
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, dummy_masks), image_pil_bboxes, gdino_conf, phrases)

            # Save the annotated image
            output_image_path = os.path.join(out_path, img_file)
            bbox_annotated_pil.save(output_image_path)

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define flag values and run the main function
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    app.run(main)
