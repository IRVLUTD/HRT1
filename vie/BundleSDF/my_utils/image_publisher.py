# my_ros_package/scripts/image_publisher.py
import rospy
from sensor_msgs.msg import Image as RosImage
import ros_numpy as rnp
import numpy as np
from PIL import Image as PILImage

class ImagePublisher:
    def __init__(self):
        self.pub_row0_rgb = rospy.Publisher("/src_rgb_with_pose", RosImage, queue_size=10)
        self.pub_row0_masked_rgb = rospy.Publisher("/src_masked_rgb", RosImage, queue_size=10)
        self.pub_row1_rgb = rospy.Publisher("/dst_rgb_with_pose", RosImage, queue_size=10)
        self.pub_row1_masked_rgb = rospy.Publisher("/dst_masked_rgb", RosImage, queue_size=10)
        self.is_first_frame = True
        self.row0_rgb = None
        self.row0_masked_rgb = None

    def publish_frames(self, frames, id_str):
        if frames is None:
            return

        try:
            # Convert PIL images to NumPy arrays and remove alpha channel
            if self.is_first_frame:
                self.row0_rgb = np.array(frames["row0"]["rgb"])[..., :3]
                self.row0_masked_rgb = np.array(frames["row0"]["masked_rgb"])[..., :3]
                self.is_first_frame = False
            row1_rgb = np.array(frames["row1"]["rgb"])[..., :3]
            row1_masked_rgb = np.array(frames["row1"]["masked_rgb"])[..., :3]

            # Convert to ROS Image messages
            ros_row0_rgb = rnp.image.numpy_to_image(self.row0_rgb, encoding="rgb8")
            ros_row0_masked_rgb = rnp.image.numpy_to_image(self.row0_masked_rgb, encoding="rgb8")
            ros_row1_rgb = rnp.image.numpy_to_image(row1_rgb, encoding="rgb8")
            ros_row1_masked_rgb = rnp.image.numpy_to_image(row1_masked_rgb, encoding="rgb8")

            # Set timestamps and frame_id
            stamp = rospy.Time.now()
            for msg in [ros_row0_rgb, ros_row0_masked_rgb, ros_row1_rgb, ros_row1_masked_rgb]:
                msg.header.stamp = stamp
                msg.header.frame_id = "camera_frame"

            # Publish
            self.pub_row0_rgb.publish(ros_row0_rgb)
            self.pub_row0_masked_rgb.publish(ros_row0_masked_rgb)
            self.pub_row1_rgb.publish(ros_row1_rgb)
            self.pub_row1_masked_rgb.publish(ros_row1_masked_rgb)
            rospy.loginfo(f"Published images for frame {id_str}")
        except Exception as e:
            rospy.logerr(f"Error publishing images: {e}")