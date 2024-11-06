# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

from absl import app, logging
from PIL import Image as PILImg
from iteach_toolkit.DHYOLO import DHYOLODetector


def main(argv):
    # Path to the input image and model
    model_path = argv[0]
    image_path = argv[1]

    try:
        logging.info("Initializing DHYOLODetector")
        dhyolo = DHYOLODetector(model_path)

        logging.info("Opening the image and converting to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")

        logging.info("Performing prediction")
        orig_image, detections = dhyolo.predict(image_path, conf_thres=0.7, iou_thres=0.7, max_det=1000)

        # Plot the bounding boxes on the original image
        orig_image, image_with_bboxes = dhyolo.plot_bboxes(attach_watermark=True)

        # Convert the image (with bounding boxes) from a NumPy array to a PIL Image for display.
        pil_img_with_bboxes = PILImg.fromarray(image_with_bboxes)

        # Show the image with bounding boxes
        pil_img_with_bboxes.show()

    except Exception as e:
        # Handle unexpected errors
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the main function with the model path and the input image path
    # app.run(main, ['ckpts/dhyolo/dh-yolo-exp-31-pl-1532.pt', 'imgs/color-000078.png'])
    # app.run(main, ['ckpts/dhyolo/dh-yolo-exp-31-pb-1532.pt', 'imgs/color-000019.png'])
    # app.run(main, ['ckpts/dhyolo/dh-yolo-v1-pb-ddf-524.pt', 'imgs/irvl-clutter-test.png'])  # Update paths as necessary
    app.run(main, ['ckpts/dhyolo/dh-yolo-exp27-pb-1008.pt', 'imgs/jpad-irvl-iteach.png'])  # Update paths as necessary
