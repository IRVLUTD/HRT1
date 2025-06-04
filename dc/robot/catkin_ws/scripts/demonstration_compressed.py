import os
import cv2
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage

class ImageListener:
    def __init__(self):
        # Publisher to Hololens stream only
        self.hololens_stream_pub = rospy.Publisher("/hololens_stream/compressed", CompressedImage, queue_size=1)

        # Subscribe to RGB image
        rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.callback, queue_size=2)
        rospy.sleep(5)

    def callback(self, rgb):
        self.im = ros_numpy.numpify(rgb)[:,:,::-1]
        self.header = rgb.header

    def publish_overlay_image(self):
        while not rospy.is_shutdown():
            try:
                im = self.im.copy()
                # overlay_image_raw = 0.4 * im.astype(np.float32) + 0.6 * self.rendered_image.astype(np.float32)
                # overlay_image = np.clip(overlay_image_raw, 0, 255).astype(np.uint8)[:, :, ::-1]  # RGB to BGR

                # JPEG encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                success, encoded_image = cv2.imencode('.jpg', im, encode_param)
                # success, encoded_image = cv2.imencode('.jpg', overlay_image, encode_param)
                if not success:
                    rospy.logwarn("JPEG encoding failed.")
                    continue

                # Compressed message
                compressed_msg = CompressedImage()
                compressed_msg.header = self.header
                compressed_msg.format = "jpeg"
                compressed_msg.data = encoded_image.tobytes()

                # Publish to Hololens topic
                self.hololens_stream_pub.publish(compressed_msg)
                rospy.loginfo("Published to /hololens_stream/compressed")
                rospy.sleep(0.1)
            except AttributeError:
                rospy.logwarn("Waiting for image...")
                rospy.sleep(1)
            except KeyboardInterrupt:
                rospy.loginfo("Interrupted.")
                break

if __name__ == "__main__":
    import pdb
    from matplotlib import pyplot as plt

    rospy.init_node("demo_overlay")
    listener = ImageListener()
    listener.publish_overlay_image()

