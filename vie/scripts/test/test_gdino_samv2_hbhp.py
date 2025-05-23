#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
To do:
- Take the task specific data root folder path which contains rgb, depth, pose directory
- Check whether save data script in robokit is running fine'
- Human Pose Hand Boxes give human pose and hand bboxes (right and left hand_bboxes)
- GDINO "objects" will give object specific bboxes (obj_bboxes)
- Feed gdino obj_bboxes to SAMV2 to track the object
- Feed rh_bbox, lh_bbox to track the both the hands
- The user knows which is the object and which is the hand that does the task
- Left to the user for post processing
"""

import os
import cv2
import numpy as np
from PIL import Image as PILImg
from absl import app, logging, flags
from HumanPoseHandBoxes import MPLandmarking as MPL, OutputImages as OUT
from robokit.perception import GroundingDINOObjectPredictor, SAM2VideoPredictor

# Define flags for data root and video directory
FLAGS = flags.FLAGS
flags.DEFINE_string('data_root', '', 'Path to the directory containing rgb, depth, and pose folders')


class VIEObjectProcessor:
    def __init__(self):
        """
        Initializes the models (GroundingDINO and SAM2) once, so they don't need to be initialized repeatedly.
        """
        logging.info("Initializing models...")

        # Initialize GroundingDINOObjectPredictor once
        self.gdino = GroundingDINOObjectPredictor()

        # Initialize SAM2VideoPredictor once
        self.sam2 = SAM2VideoPredictor()

        logging.info("Models initialized.")

    def process_image(self, data_root, image_file, text_prompt="objects"):
        """
        Processes a single image with human pose and hand bounding boxes, and GDINO object detection.

        Args:
            data_root (str): Root directory containing 'rgb', 'depth', and 'pose' subdirectories.
            image_file (str): Filename of the image in the 'rgb' directory.
            text_prompt (str): Text prompt for GroundingDINO to detect specific objects.

        Returns:
            dict: A dictionary containing image_pil_bboxes, imageWithBB, leftHandImage, rightHandImage, rBB, and lBB.
        """
        image_path = os.path.join(data_root, "rgb", image_file)
        
        try:
            # Open the image and convert to RGB format for GDINO
            image_pil = PILImg.open(image_path).convert("RGB")

            logging.info("GDINO: Predicting bounding boxes and confidence scores")
            bboxes, phrases, gdino_conf = self.gdino.predict(image_pil, text_prompt)

            logging.info("GDINO post-processing for bounding boxes")
            w, h = image_pil.size  # Image dimensions
            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            # Load image using OpenCV for hand bounding box processing
            image_cv = cv2.imread(image_path)

            # Use MPL to get human pose landmarks
            logging.info("Extracting human pose landmarks")
            landmarks = MPL.mediaPipeOnImageFilePath(image_path)

            # Process image with bounding boxes for hands
            imageWithBB, leftHandImage, rightHandImage, rBB, lBB = OUT.getImageWithBB(image_cv, landmarks, 0.7)

            return {
                "human": {
                    "imageWithBB": imageWithBB,
                    "leftHandImage": leftHandImage,
                    "rightHandImage": rightHandImage,
                    "rBB": rBB,
                    "lBB": lBB
                },
                "gdino": {
                    "bboxes": image_pil_bboxes,
                    "phrases": phrases,
                    "conf": gdino_conf
                },
            }

        except Exception as e:
            logging.error(f"An error occurred while processing {image_file}: {e}")
            return None

    def process_video(self, video_dir, bbox=np.array([224, 155, 250, 160]), propagation_interval=2):
        """
        Processes a video directory using SAM2 for segmentation and mask propagation.

        Args:
            video_dir (str): Directory containing video frames in JPG format.
            bbox (np.array): Bounding box for SAM2 initialization.
            propagation_interval (int): Interval for mask propagation.

        Returns:
            dict: A dictionary containing frame names and video segments.
        """
        try:
            logging.info("SAM2: Propagating masks and saving results")
            frame_names, video_segments = self.sam2.propagate_masks_and_save(video_dir, bbox, propagation_interval)
            
            return {
                "frame_names": frame_names,
                "video_segments": video_segments
            }

        except Exception as e:
            logging.error(f"An error occurred while processing video in {video_dir}: {e}")
            return None


def main(argv):
    """
    Main function to process each image in the RGB directory and video frames in the specified video directory.

    Args:
        argv (list): List of arguments passed to the script. Expects data root and video directory paths.
    """
    data_root = FLAGS.data_root # Data root for images
    video_dir = os.path.join(data_root, "rgb")  # Video directory for SAM2

    # Create an instance of the VIEObjectProcessor class
    processor = VIEObjectProcessor()

    # Process all images in the RGB directory
    rgb_path = os.path.join(data_root, "rgb")
    for image_file in os.listdir(rgb_path):
        logging.info(f"Processing image: {image_file}")
        result = processor.process_image(data_root, image_file)
        
        if result:
            # Save or further process the results from `result` dictionary as needed
            # it's already in pixel xyxy coordinates format
            print(result)
            print(f"Processed results for {image_file}: {result.keys()}")  # Replace this with saving logic
            
    # Process video frames in the specified directory
    video_result = processor.process_video(video_dir)
    if video_result:
        # Save or further process the results from `video_result` dictionary as needed
        print(f"Processed video frames: {video_result['frame_names']}, Segments: {len(video_result['video_segments'])}")

if __name__ == "__main__":
    # Define flags
    flags.mark_flag_as_required('data_root')

    # Run the main function with data root and video directory paths
    app.run(main)
