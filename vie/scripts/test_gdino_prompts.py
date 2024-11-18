#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


"""
This script performs object detection on a set of input images using Grounding DINO. 
It annotates each image with detected bounding boxes, confidence scores, and labels based on a text prompt, 
and saves the results in a specified output directory.

The script follows these steps:

1. **Initialize Flags**:
   - Define command-line flags for the input directory (`input_dir`) containing images and a text prompt (`text_prompt`) 
     for guiding the object detection with Grounding DINO.

2. **Main Function**:
   - Parse the command-line flags to obtain the input directory and text prompt.
   - Ensure both `input_dir` and `text_prompt` are provided; otherwise, raise an error.

3. **Setup Object Detector**:
   - Initialize the Grounding DINO object detector for identifying bounding boxes and associated labels based on the text prompt.

4. **Set Output Directory**:
   - Construct the output directory path as a new subdirectory named `gdino/<text_prompt>` 
     (text prompt is converted to lowercase and spaces are replaced with underscores) within the parent directory of `input_dir`.
   - Create the output directory if it doesn’t exist.

5. **Process Each Image**:
   - For each image in `input_dir`:
     a. Load and convert the image to RGB format.
     b. Use Grounding DINO to predict bounding boxes, phrases, and confidence scores for the image based on the text prompt.
     c. Scale the bounding boxes to match the original image dimensions.
     d. Annotate the image with bounding boxes, confidence scores, and labels.

6. **Save Annotated Image**:
   - Save the annotated image in the output directory, preserving the original image filename.

7. **Error Handling**:
   - Print an error message if any unexpected errors occur.

**Usage**:
   - Run this script from the command line, passing the required `input_dir` and `text_prompt` flags:
   
     `python test_gdino_prompts.py --input_dir=<path_to_images> --text_prompt="object description"`
"""


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
    _image_root_dir, text_prompt = sanity_check(argv)

    if not _image_root_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()

        # Set output directory in the parent directory of _image_root_dir
        parent_dir = os.path.dirname(_image_root_dir)
        out_path_suffix = f"out/gdino/{text_prompt.lower().replace(' ', '_')}"
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



def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt
    
    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(f"Error: The directory '{input_dir}' does not exist or is not a valid directory.")
        

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise Exception(f"No image files found in the directory '{input_dir}'.")
    
    # Ensure text_prompt is provided
    if not text_prompt:
        raise Exception("Error: 'text_prompt' is required but not provided.")

    return input_dir, text_prompt


if __name__ == "__main__":
    # Define flag values and run the main function
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    app.run(main)
