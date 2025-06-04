import os
import cv2
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage

class ImageListener:
    def __init__(self, rendered_image):
        # Publisher to Hololens stream only
        self.hololens_stream_pub = rospy.Publisher("/hololens_stream/compressed", CompressedImage, queue_size=1)
        self.rendered_image = rendered_image

        # Subscribe to RGB image
        rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.callback, queue_size=2)
        rospy.sleep(5)

    def callback(self, rgb):
        self.im = ros_numpy.numpify(rgb)
        self.header = rgb.header

    def publish_overlay_image(self):
        while not rospy.is_shutdown():
            try:
                im = self.im.copy()
                overlay_image_raw = 0.4 * im.astype(np.float32) + 0.6 * self.rendered_image.astype(np.float32)
                overlay_image = np.clip(overlay_image_raw, 0, 255).astype(np.uint8)[:, :, ::-1]  # RGB to BGR

                # JPEG encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                success, encoded_image = cv2.imencode('.jpg', overlay_image, encode_param)
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
    root_dir = "./render_data"

    object_folders = os.listdir(root_dir)
    for index, object_folder in enumerate(object_folders):
        print(f"{index}: {object_folder}")
    user_choice = int(input("Enter desired object index: "))
    desired_object = object_folders[user_choice]

    rendered_image_name = "render_" + desired_object + ".png"
    rendered_image_path = os.path.join(root_dir, desired_object, rendered_image_name)
    rendered_image = cv2.imread(rendered_image_path)[:, :, ::-1]  # Convert BGR to RGB

    listener = ImageListener(rendered_image)
    listener.publish_overlay_image()

