#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------


import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


def image_callback(msg):
    """Callback to handle the image subscription, compression and publish."""
    bridge = CvBridge()

    # Convert the ROS Image message to a OpenCV image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("Error converting image: %s", e)
        return
    
    # Compress the image using OpenCV
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # You can adjust the quality (0-100)
    _, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)

    # Create a CompressedImage message
    compressed_img_msg = CompressedImage()
    compressed_img_msg.header = msg.header
    compressed_img_msg.format = "jpeg"
    compressed_img_msg.data = encoded_image.tobytes()

    # Publish the compressed image to the new topic
    compressed_image_pub.publish(compressed_img_msg)
    rospy.loginfo("Published compressed image.")

def image_subscriber():
    """Initialize the image subscriber and publisher."""
    rospy.init_node('image_compressor', anonymous=True)

    # Subscribe to the raw image topic
    rospy.Subscriber('/head_camera/rgb/image_raw', Image, image_callback)

    # Publish compressed image to a new topic
    global compressed_image_pub
    compressed_image_pub = rospy.Publisher('/head_camera/rgb/image_raw/compressed', CompressedImage, queue_size=10)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        image_subscriber()
    except rospy.ROSInterruptException:
        pass
