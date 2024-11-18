from .detect import run as run_detection
import torch
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DHYOLODetector:
    """
    A class to encapsulate the YOLO model prediction for object detection.

    Attributes:
        model_path (str): Path to the trained YOLO model weights.

    Methods:
        predict(image_path, conf_thres, iou_thres, max_det):
            Runs object detection on the given image using the YOLO model.
            Logs errors in case of failure and handles exceptions.
        plot_bboxes():
            Plots bounding boxes on the input image based on YOLO predictions and returns the modified image.
    """
    
    def __init__(self, model_path):
        """
        Initializes DHYOLODetector with the path to the YOLO model weights.
        
        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model_path = model_path
        self.image = None
        self.preds = None

    def predict(self, image_path, conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """
        Runs object detection on the provided image using the YOLO model.
        
        Args:
            image_path (str): Path to the image file (file path or a web URL).
            conf_thres (float): Confidence threshold for YOLO detections (default is 0.25).
            iou_thres (float): IOU threshold for non-max suppression (default is 0.45).
            max_det (int): Maximum number of detections per image (default is 1000).
        
        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The image read as a NumPy array.
                - dict: A dictionary with the bounding boxes in xyxy format, confidence scores, and class labels:
                    - 'boxes': List of bounding boxes in xyxy format.
                    - 'confidences': List of confidence scores for each detection.
                    - 'class_labels': List of class labels corresponding to each detection.
        
        Raises:
            FileNotFoundError: If the image file does not exist or cannot be opened.
            Exception: For any general errors during prediction.
        """
        try:
            logger.info(f"Starting prediction for image: {image_path} with model: {self.model_path}")

            
            self.image = cv2.imread(image_path) # Read the image as a NumPy array
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

            if self.image is None:
                raise FileNotFoundError(f"Image file not found or cannot be opened: {image_path}")

            # Run YOLO detection with custom parameters
            result = run_detection(
                weights=self.model_path,
                source=image_path,
                conf_thres=conf_thres,  # Confidence threshold
                iou_thres=iou_thres,    # IOU threshold
                max_det=max_det          # Maximum detections
            )

            self.preds = result[0]  # We are only running for one image

            # Prepare predictions as a dictionary
            detections_dict = {
                'boxes': [],          # List of bounding boxes in xyxy format
                'confidences': [],    # List of confidence scores
                'class_labels': []     # List of class labels
            }

            for detection in self.preds:
                if isinstance(detection, torch.Tensor):
                    detection = detection.cpu().numpy()

                # Extract details
                x1, y1, x2, y2, conf, cls = detection[:6]
                detections_dict['boxes'].append([float(x1), float(y1), float(x2), float(y2)])  # xyxy format
                detections_dict['confidences'].append(float(conf))
                detections_dict['class_labels'].append(int(cls))

            logger.info(f"Prediction completed successfully for image: {image_path}")
            return self.image, detections_dict  # Return image and detections as a tuple
        
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {image_path}. Exception: {e}")
            raise
        
        except Exception as e:
            logger.error(f"An error occurred during prediction. Exception: {e}")
            raise

    def plot_bboxes(self, attach_watermark=False):
        """
        Plots bounding boxes on the input image based on YOLO predictions and returns the modified image.

        Args:
            attach_watermark (bool): Whether to attach a watermark text to the image (default is False).

        Returns:
            tuple: The original image and the image with bounding boxes plotted.
        """
        class_labels = {0: "door", 1: "handle"}
        class_colors = {
            0: (255, 0, 0),  # Red in RGB format for doors
            1: (255, 255, 0)  # Yellow in RGB format for handles
        }

        bbox_img = self.image.copy()  # Create a copy of the original image

        # Check if there are predictions
        if self.preds is None or len(self.preds) == 0:
            logger.warning("No predictions to display.")
            return bbox_img, bbox_img  # Return the original image if no predictions

        # Iterate through detections and plot each bounding box
        for detection in self.preds:
            if isinstance(detection, torch.Tensor):
                detection = detection.cpu().numpy()

            conf = detection[4]
            x1, y1, x2, y2, _, cls = detection[:6].astype(float)  # Ensure float for bounding box coordinates
            label = class_labels[int(cls)]

            # Draw the rectangle on the bbox_img
            cv2.rectangle(bbox_img, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[int(cls)], 2)

            # Prepare text with confidence score
            text = f'{label} ({conf:.2f})'  # Include confidence score in the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]

            # Set text position directly above the bounding box
            text_x = int(x1)
            text_y = int(y1) - 2  # Adjust for a slight overlap with the bounding box

            # Set text color based on class
            text_color = (0, 0, 0) if cls == 1 else (255, 255, 255)  # Black for handle, white for door

            # Draw a background rectangle for the text
            cv2.rectangle(bbox_img, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y), class_colors[int(cls)], cv2.FILLED)

            # Put the label text on the bbox_img
            cv2.putText(bbox_img, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Attach watermark if specified
        if attach_watermark:
            watermark_text = "Predictions by DH-YOLO"
            watermark_color = (200, 200, 200)  # Greyish color for watermark
            watermark_scale = 0.4  # Reduced scale for the watermark text
            watermark_thickness = 1  # Decreased thickness for the watermark text
            
            # Get the text size for positioning
            text_size = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_DUPLEX, watermark_scale, watermark_thickness)[0]
            text_x = bbox_img.shape[1] - text_size[0] - 10  # 10 pixels from right
            text_y = bbox_img.shape[0] - 10  # 10 pixels from bottom
            
            # Put the watermark text on the bbox_img
            cv2.putText(bbox_img, watermark_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, watermark_scale, watermark_color, watermark_thickness, cv2.LINE_AA)

        return self.image, bbox_img  # Return the original image and modified image with bounding boxes
