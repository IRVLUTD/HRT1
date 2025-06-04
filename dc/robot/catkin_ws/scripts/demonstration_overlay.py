#!/usr/bin/env python
import sys, os
import cv2
import scipy.io
import rospy
import numpy as np
import tf

from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
import ros_numpy


class ImageListener:
    def __init__(self, rendered_image):
        
        # pose image publisher
        self.overlay_img_publisher = rospy.Publisher("image_overlay", Image, queue_size=1)
        self.rendered_image = rendered_image

        rgb_sub = rospy.Subscriber(
            "/head_camera/rgb/image_raw", Image, self.callback, queue_size=2
        )
        rospy.sleep(5)

   

    def callback(self, rgb):
        # get color images
        self.im = ros_numpy.numpify(rgb)

    
    def publish_overlay_image(self):

        while True:
            try:
                im = self.im.copy()
                overlay_image_raw = 0.4 * im.astype(np.float32) + 0.6 * self.rendered_image.astype(
                np.float32)
                overlay_image = np.clip(overlay_image_raw, 0, 255).astype(np.uint8)
                overlay_image_msg = ros_numpy.msgify(Image, overlay_image, encoding='rgb8')
                    # pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
                    # pose_msg.header.stamp = rospy.Time.now()
                    # pose_msg.header.frame_id = rgb.header.frame_id
                    # pose_msg.encoding = "bgr8"
                self.overlay_img_publisher.publish(overlay_image_msg)
                print("publish reference image at /image_overlay")
            except KeyboardInterrupt:
                exit(0)

            





if __name__ == "__main__":
    import pdb
    from matplotlib import pyplot as plt
    """
    Main function to run the code
    """
    rospy.init_node("demo_overlay")
    root_dir = "./gazebo_rendered_ycb_data"
    # read a reference scene
    object_folders = os.listdir(root_dir)
    for index, object_folder in enumerate(object_folders):
        print(f"{index}: {object_folder}")
    user_choice = int(input("Enter desired object index"))
    desired_object = object_folders[user_choice]    
    rendered_image_name = "render_" + desired_object + ".png"
    rendered_image_path = f"{os.path.join(os.path.join(root_dir, desired_object), rendered_image_name)}" 

    rendered_image = cv2.imread(rendered_image_path)[:,:,::-1]

    # pdb.set_trace()
    # plt.imshow( rendered_image)
    # plt.show()
    # data = read_data(dirname, index)

    # image listener
    listener = ImageListener(rendered_image)
    listener.publish_overlay_image()
