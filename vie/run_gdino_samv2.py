# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
# ----------------------------------------------------------------------------------------------------

"""
This script performs object tracking on video frames using Grounding DINO for initial detection
and SAM2 for tracking each detected bounding box across frames.

The script follows these steps:

1. **Initialize Flags**:
   - Define command-line flags for the input directory containing video frames, a text prompt for object detection,
     and an interval for saving frames during tracking.

2. **Main Function**:
   - Parse the command-line flags to obtain the input directory, text prompt, and save interval.
   - Ensure both input directory and text prompt are provided; otherwise, raise an error.

3. **Setup Object Detectors**:
   - Initialize the Grounding DINO object detector to detect bounding boxes in the first frame based on the given text prompt.
   - Initialize SAM2 to track detected bounding boxes across all frames.

4. **Process First Frame**:
   - Load the first frame from the input directory.
   - Use Grounding DINO to detect initial bounding boxes with the specified text prompt.

5. **Track All Detected Bounding Boxes**:
   - For each bounding box detected in the first frame:
     a. Convert the bounding box to a [x_min, y_min, x_max, y_max] format.
     b. Pass each bounding box to SAM2 to propagate the bounding box across all frames, tracking the object's movement.
     c. Set the tracking save interval to the specified value for periodic saving of the tracking results.

6. **Output**:
   - Each bounding box's tracking results are saved in the specified save interval.
   - Log messages indicate the start and completion of tracking for each bounding box.

**Usage**:
   - Run this script from the command line, passing the required `input_dir` and `text_prompt` flags,
     and an optional `save_interval` (default is 1):

     `python test_gdino_samv2.py --input_dir=<path_to_frames> --text_prompt="object description" --save_interval=2`
"""


import os
import numpy as np
from absl import app, flags, logging
from PIL import Image as PILImg
from robokit.perception import GroundingDINOObjectPredictor, SAM2VideoPredictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory path to input video frames")
flags.DEFINE_string("text_prompt", None, "Text prompt for initial object detection")
flags.DEFINE_integer("save_interval", 1, "Interval for saving tracked frames")


def main(argv):
    # Get input values from flags
    video_dir, text_prompt = sanity_check(argv)
    save_interval = FLAGS.save_interval

    # Check if required flags are provided
    if not video_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    try:
        logging.info("Initialize object detectors")

        # Initialize Grounding DINO for initial bbox detection
        gdino = GroundingDINOObjectPredictor()

        # Initialize SAM2 for tracking across frames
        sam2 = SAM2VideoPredictor(text_prompt)

        # Read the first frame to detect initial bounding boxes
        first_frame_path = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
        first_frame = PILImg.open(first_frame_path).convert("RGB")

        logging.info(
            "GDINO: Predict initial bounding boxes, phrases, and confidence scores"
        )
        initial_bboxes, _, _ = gdino.predict(first_frame, text_prompt)

        # Track each bounding box across frames
        if len(initial_bboxes) > 0:
            logging.info(
                f"Detected {len(initial_bboxes)} bounding boxes in the first frame"
            )

            # Process each bounding box
            for i, bbox in enumerate(initial_bboxes):
                # Convert bbox to [x_min, y_min, x_max, y_max] format
                x_min, y_min, x_max, y_max = gdino.bbox_to_scaled_xyxy(
                    bbox, *first_frame.size
                )
                bbox_array = np.array([x_min, y_min, x_max, y_max])

                logging.info(f"SAM2: Track bounding box {i+1} across all frames")
                # Track and propagate the bounding box across frames with the specified save interval
                frame_names, video_segments = sam2.propagate_masks_and_save(
                    video_dir, bbox_array, save_interval
                )

            logging.info("Tracking complete for all bounding boxes")
        else:
            logging.error("No bounding boxes detected in the first frame")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt

    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(
            f"Error: The directory '{input_dir}' does not exist or is not a valid directory."
        )

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise Exception(f"No image files found in the directory '{input_dir}'.")

    # Ensure text_prompt is provided
    if not text_prompt:
        raise Exception("Error: 'text_prompt' is required but not provided.")

    return input_dir, text_prompt


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required("input_dir")
    flags.mark_flag_as_required("text_prompt")

    # Run the main function
    app.run(main)
