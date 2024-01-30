from absl import app
from PIL import Image as PILImg
from robokit.utils import annotate
from robokit.ObjDetection import GroundingDINOObjectDetector, ZeroShotClipPredictor

def main(argv):
    # Path to the input image
    image_path = argv[0]
    # List of text prompts
    text_prompts =  ['mug', 'power drill', 'cellphone', 'banana', 'none']
    
    try:
        # Initialize object detectors
        gdino = GroundingDINOObjectDetector()
        _clip = ZeroShotClipPredictor()

        # Open the image and convert to RGB format
        image_pil = PILImg.open(image_path).convert("RGB")
        
        # Predict bounding boxes, phrases, and confidence scores
        bboxes, phrases, gdino_conf = gdino.predict(image_pil)

        # Get image width and height
        w, h = image_pil.size
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        # Crop images based on bounding boxes
        cropped_bbox_imgs = list(map(lambda bbox: (image_pil.crop(bbox.int().numpy())), image_pil_bboxes))
        # Use ZeroShotClipPredictor to predict labels for cropped images
        clip_conf, idx = _clip.predict(cropped_bbox_imgs, text_prompts)
        
        # Scale the original image for visualization
        scaled_image_pil = gdino.image_transform_for_vis(image_pil)
        # Get the size of the scaled image
        ws, hs = scaled_image_pil.size
        # Scale bounding boxes to match the scaled image size
        scaled_image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, ws, hs)
        # Get the predicted labels based on the indices
        predicted_labels = [ text_prompts[i] for i in idx ]
        # Annotate the scaled image with bounding boxes, confidence scores, and labels, and display
        annotate(scaled_image_pil, scaled_image_pil_bboxes, clip_conf, predicted_labels).show()


    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the main function with the input image path
    app.run(main, ['imgs/irvl-clutter-test.png.png'])
