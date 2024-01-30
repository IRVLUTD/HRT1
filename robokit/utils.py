import os
import numpy as np
import supervision as sv
from PIL import Image as PILImg


def file_exists(file_path):
    """
    Check if a file exists.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)


def crop_images(original_image, bounding_boxes):
    """
    Crop the input image using the provided bounding boxes.

    Parameters:
    - original_image (PIL.Image.Image): Original input image.
    - bounding_boxes (list): List of bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
    - cropped_images (list): List of cropped images.

    Raises:
    - ValueError: If the bounding box dimensions are invalid.
    """
    cropped_images = []

    try:
        for box in bounding_boxes:
            if len(box) != 4:
                raise ValueError("Bounding box should have 4 values: [x_min, y_min, x_max, y_max]")

            x_min, y_min, x_max, y_max = box

            # Check if the bounding box dimensions are valid
            if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                raise ValueError("Invalid bounding box dimensions")

            # Crop the image using the bounding box
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped_image)

    except ValueError as e:
        print(f"Error in crop_images: {e}")


def annotate(image_source, boxes, logits, phrases):
    detections = sv.Detections(xyxy=boxes.numpy())
    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]
    box_annotator = sv.BoxAnnotator()
    img_pil = PILImg.fromarray(box_annotator.annotate(scene=np.array(image_source), detections=detections, labels=labels))
    return img_pil

