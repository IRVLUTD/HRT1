#!/usr/bin/env python

import sys
import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from mask_service.srv import GetMask, GetMaskRequest
import ros_numpy

class MaskServiceClient:
    """
    Client for calling the GetMask service to get segmentation masks.
    """
    def __init__(self):

        self.service_name = 'get_mask'
        rospy.wait_for_service(self.service_name, timeout=10.0)
        self.service_proxy = rospy.ServiceProxy(self.service_name, GetMask)

    def call_mask_service(self, rgb_image_path="", text_prompt=""):
        """
        calls the GetMask service with an RGB image and text prompt
        Arg: rgb_image_path if None, service will query listener. else it will readf rom the path
             text_prompt
        """
        try:
            req = GetMaskRequest()
            req.textprompt = String(data=text_prompt)

            if rgb_image_path and os.path.exists(rgb_image_path):
                # Read and convert image
                rgb_img = cv2.imread(rgb_image_path)
                if rgb_img is None:
                    rospy.loginfo(f"failed to read image at: {rgb_image_path}")
                    return None
                rgb_img = rgb_img[:,:,::-1]  # Convert BGR to RGB
                rgb_msg = ros_numpy.msgify(Image, rgb_img, 'rgb8')
                rgb_msg.header.frame_id = "head_camera_rgb_optical_frame"
                rgb_msg.header.stamp = rospy.Time.now()
                req.rgb = rgb_msg
            else:
                # Send empty image
                req.rgb = Image()
                req.rgb.header.frame_id = "head_camera_rgb_optical_frame"
                req.rgb.header.stamp = rospy.Time.now()
                req.rgb.height = 0
                req.rgb.width = 0
                req.rgb.encoding = 'rgb8'
                req.rgb.data = []

            # Call service
            response = self.service_proxy(req)
            if response.mask.height == 0 or response.mask.width == 0:
                rospy.loginfo("Received empty mask from service")
                return None

            mask = ros_numpy.numpify(response.mask)
            return mask

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
        except Exception as e:
            rospy.logerr(f"Error processing image or response: {e}")
            return None

    def save_and_display_mask(self, mask, output_path):
        if mask is not None:
            mask[mask > 0] = 255  # Convert to binary mask
            cv2.imshow("Mask", mask)
            cv2.waitKey(0)
            cv2.imwrite(output_path, mask)
            rospy.loginfo(f"Mask saved to {output_path}")
        else:
            rospy.loginfo("No mask to save or display")

if __name__ == "__main__":
    try:
        rospy.init_node('mask_service_client', anonymous=True)
        if len(sys.argv) != 4:
            print("Usage: python mask_service_client.py <image_path> <text_prompt> <output_mask_path>")
            sys.exit(1)

        image_path = sys.argv[1]
        text_prompt = sys.argv[2]
        output_mask_path = sys.argv[3]

        client = MaskServiceClient()
        rospy.sleep(1)  

        mask = client.call_mask_service(image_path, text_prompt)

        client.save_and_display_mask(mask, output_mask_path)

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")