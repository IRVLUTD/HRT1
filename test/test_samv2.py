#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import numpy as np
from absl import app, logging
from robokit.perception import SAM2VideoPredictor


def main(argv):

    video_dir = argv[0]

    try:
        logging.info("Initialize object detectors")
        sam2 = SAM2VideoPredictor()

        logging.info("SAM2 prediction")
        # sam2.segment_using_bbox(video_dir, 0, np.array([224,155, 250,160]))
        frame_names, video_segments = sam2.propagate_masks_and_save(video_dir, np.array([224,155, 250,160]), 2)
    
    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the main function with the input video dir path containing jpg frame images
    app.run(
        main, 
        ['imgs/irvl-whiteboard-write-and-erase']
    )
