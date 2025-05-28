#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from PIL import Image as PILImg
from sensor_msgs.msg import Image
from mask_service.srv import GetMask, GetMaskResponse
from listener import ImageListener
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import ros_numpy

class MaskService:
    def __init__(self):
        # Initialize GDINO and SAM models
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.listener = ImageListener()

    def run_network(self, rgb_image, text_prompt, frame_id, stamp):
        """Run the network to generate a mask."""
        # BGR image
        im = rgb_image.astype(np.uint8)
        img_pil = PILImg.fromarray(im)

        # Run GroundingDINO and SAM
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, text_prompt)
        if phrases == []:
            rospy.loginfo(f"No object detected by GDINO for the prompt -  {text_prompt}")
            return None
        w, h = im.shape[1], im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
        masks = masks[index]
        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]


        # Convert mask to sensor_msgs/Image
        mask_msg = ros_numpy.msgify(Image, mask.astype(np.uint8), 'mono8')
        mask_msg.header.stamp = stamp
        mask_msg.header.frame_id = frame_id
        mask_msg.encoding = 'mono8'

        return mask_msg

    def handle_get_mask(self, req):
        """Handle GetMask service request."""
        rospy.loginfo(f"Received GetMask request with prompt: {req.textprompt.data}")
        text_prompt = req.textprompt.data

        # 1. If i send empty request, then grab the image froim listener
        # 2. If i send an image in the request, evaluate it
        if req.rgb.height == 0 or req.rgb.width == 0:
            rgb_image, frame_id, stamp = self.listener.get_image()
            if rgb_image is None:
                rospy.logerr("Image from listener is None! sending empty mask response")
                return GetMaskResponse()
        else:
            rgb_image = ros_numpy.numpify(req.rgb)
            frame_id = req.rgb.header.frame_id
            stamp = req.rgb.header.stamp

        mask_msg = self.run_network(rgb_image, text_prompt, frame_id, stamp)
        if mask_msg is None:
            rospy.logerr("No mask generated! returning empty mask response")
            return GetMaskResponse()
        return GetMaskResponse(mask=mask_msg)

def mask_service_server():
    rospy.init_node('mask_service_server')
    mask_service = MaskService()
    s = rospy.Service('get_mask', GetMask, mask_service.handle_get_mask)
    rospy.loginfo("Mask service server ready")
    rospy.spin()

if __name__ == "__main__":
    try:
        mask_service_server()
    except rospy.ROSInterruptException:
        pass