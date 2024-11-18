import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import os

# Set the base path to the directory where the script is located
this_file_base_path = os.path.dirname(__file__)

"""draws all pose landmarks on an image"""
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

"""Given a filepath to an image returns the detection results"""
def mediaPipeOnImageFilePath(filePath):
  base_options = python.BaseOptions(model_asset_path=os.path.join(this_file_base_path, 'pose_landmarker_heavy.task'))
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)

  detector = vision.PoseLandmarker.create_from_options(options)
  image = mp.Image.create_from_file(filePath)
  detection_result = detector.detect(image)

  return detection_result
