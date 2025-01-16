#!/usr/bin/env python

"""
RUN: python ros_test_images.py ./_DATA/new-data-from-fetch-and-laptop/22tasks/task_15_19s-use-knife/ "knife"
"""

import os
import cv2
import sys
import time
import rospy
import shutil
import threading
import ros_numpy
import subprocess
import numpy as np
import message_filters
from PIL import Image as PILImg
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor

lock = threading.Lock()

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self, text_prompt, camera='Fetch'):

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.text_prompt = text_prompt
        self.trigger_flag = None
    
        # initialize network
        self.text_prompt =  ''          
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()     

        # initialize a node
        rospy.init_node("seg_rgb")
        self.label_pub = rospy.Publisher('seg_label_refined', Image, queue_size=10)
        self.score_pub = rospy.Publisher('seg_score', Image, queue_size=10)     
        self.image_pub = rospy.Publisher('seg_image', Image, queue_size=10)

        rospy.Subscriber('/collect_mask_for_opt', String, self.trigger_for_save_pre_post_base_opt_mask_img)

        if camera  == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Realsense':
            # use RealSense D435
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (camera)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (camera), Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (camera), Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (camera), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (camera)
            self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0
            depth_cv = depth_cv * 1000
            depth_cv = depth_cv.astype(np.uint16)

        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_network(self):

        with lock:
            if self.im is None:
                return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            rospy.sleep(1)
        print('===========================================')

        # bgr image
        im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]
        img_pil = PILImg.fromarray(im)
        # self.text_prompt = text_prompt
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)

        # Scale bounding boxes to match the original image size
        w = im.shape[1]
        h = im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        # logging.info("SAM prediction")
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)

        # filter large boxes
        image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
        masks = masks[index]
        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]

        # logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
        # bbox_annotated_pil.show()
        im_label = np.array(bbox_annotated_pil)

        # show result
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # plt.imshow(im_label)
        # ax.set_title('output image')
        # ax = fig.add_subplot(1, 2, 2)
        # plt.imshow(mask)
        # ax.set_title('mask')              
        # plt.show()        

        # publish segmentation mask
        label = mask
        self.mask=mask
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)

        # publish score map
        score = label.copy()
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        for index, mask_id in enumerate(mask_ids):
            score[label == mask_id] = gdino_conf[index]
        label_msg = ros_numpy.msgify(Image, score.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.score_pub.publish(label_msg)        

        num_object = len(np.unique(label)) - 1
        print('%d objects' % (num_object))

        # publish segmentation images
        rgb_msg = ros_numpy.msgify(Image, im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

        return self.im, self.depth, self.mask
    


    def trigger_for_save_pre_post_base_opt_mask_img(self, msg):
        self.trigger_flag = msg.data


    def save_mask(self, filename, img, depth, mask):
        # File paths
        rgb_path = os.path.join("/tmp", "{}_rgb.jpg".format(filename))
        depth_path = os.path.join("/tmp", "{}_depth.png".format(filename))
        mask_path = os.path.join("/tmp", "{}_mask.png".format(filename))

        cv2.imwrite(depth_path, depth)
        mask[mask > 0] = 255
        cv2.imwrite(mask_path, mask)
        
        PILImg.fromarray(img).save(rgb_path)



def draw_pose_axis(image, pose_matrix, intrinsic_matrix, axis_length=0.1, thickness=2, color_scheme=None):
    """
    Draws a 3D pose axis on the image based on the pose matrix.

    Args:
        image (np.ndarray): The RGB image to draw on.
        pose_matrix (np.ndarray): The 4x4 transformation matrix (pose).
        intrinsic_matrix (np.ndarray): The camera intrinsic matrix.
        axis_length (float): Length of the axis in world units.
        thickness (int): Thickness of the axis lines.
        color_scheme (tuple): Optional tuple of colors for the axes.
    
    Returns:
        np.ndarray: The image with the pose axis drawn.
    """
    # Default axis colors: (X: Red, Y: Green, Z: Blue)
    colors = color_scheme or [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Define axis points (X, Y, Z) in the object frame
    axis_points = np.array([
        [0, 0, 0, 1],             # Origin
        [axis_length, 0, 0, 1],   # X-axis
        [0, axis_length, 0, 1],   # Y-axis
        [0, 0, axis_length, 1]    # Z-axis
    ]).T  # Shape: (4, 4)

    # Transform axis points to the camera frame using the pose matrix
    transformed_points = pose_matrix @ axis_points

    # Project the 3D points to 2D image space
    points_2d = intrinsic_matrix @ transformed_points[:3, :]
    points_2d /= points_2d[2]  # Normalize by depth (z-coordinate)

    # Extract pixel coordinates
    points_2d = points_2d[:2].T.astype(int)  # Shape: (4, 2)

    # Draw axes on the image
    origin = tuple(points_2d[0])  # Origin
    for i, color in enumerate(colors):  # Draw X, Y, Z axes
        image = cv2.line(image, origin, tuple(points_2d[i + 1]), color, thickness)

    return image


def pre_post_mask_capture(listener):
    while listener.trigger_flag is None:
        continue
    
    img, depth, mask = listener.run_network()
    listener.save_mask(listener.trigger_flag, img, depth, mask)
    listener.trigger_flag = None




if __name__ == '__main__':
    # image listener
    task_dir, text_prompt = sys.argv[1], sys.argv[2]
    listener = ImageListener(text_prompt=text_prompt)
    rospy.sleep(3)

    # Define paths
    root_dir = os.path.dirname(os.path.normpath(task_dir))
    realworld_dir = "realworld"
    rgb_source_dir = os.path.join(task_dir, "rgb")
    depth_source_dir = os.path.join(task_dir, "depth")
    # text_prompt = os.listdir(os.path.join(task_dir, "out", "samv2"))[0].replace('_',' ')
    sam2_out_dir = os.path.join(task_dir, "out", "samv2")
    mask_source_dir = os.path.join(sam2_out_dir, os.listdir(sam2_out_dir)[-1], "obj_masks")
    realworld_rgb_dir = os.path.join(realworld_dir, "rgb")
    realworld_depth_dir = os.path.join(realworld_dir, "depth")
    realworld_mask_dir = os.path.join(realworld_dir, "masks")
    cam_k_file_source = os.path.join(root_dir, "cam_K.txt")
    cam_k_file_target = os.path.join(realworld_dir, "cam_K.txt")

    # Create the realworld directory structure
    os.makedirs(realworld_rgb_dir, exist_ok=True)
    os.makedirs(realworld_depth_dir, exist_ok=True)
    os.makedirs(realworld_mask_dir, exist_ok=True)

    # Select and copy the first frame (assumed to be 000000.jpg)
    first_frame = "000000.jpg"
    first_frame_source = os.path.join(rgb_source_dir, first_frame)
    first_frame_target = os.path.join(realworld_rgb_dir, first_frame)

    if os.path.exists(first_frame_source):
        shutil.copy(first_frame_source, first_frame_target)
        print(f"Copied {first_frame_source} to {first_frame_target}")
    else:
        print(f"First frame {first_frame_source} does not exist.")

    # Select and copy the depth of the first frame
    first_frame_depth = "000000.png"  # Assuming mask files are PNG with matching names
    first_frame_depth_source = os.path.join(depth_source_dir, first_frame_depth)
    first_frame_depth_target = os.path.join(realworld_depth_dir, first_frame_depth)

    if os.path.exists(first_frame_depth_source):
        shutil.copy(first_frame_depth_source, first_frame_depth_target)
        print(f"Copied {first_frame_depth_source} to {first_frame_depth_target}")
    else:
        print(f"Mask for the first frame {first_frame_depth_source} does not exist.")


    # Select and copy the mask of the first frame
    first_frame_mask = "000000.png"  # Assuming mask files are PNG with matching names
    first_frame_mask_source = os.path.join(mask_source_dir, first_frame_mask)
    first_frame_mask_target = os.path.join(realworld_mask_dir, first_frame_mask)

    if os.path.exists(first_frame_mask_source):
        shutil.copy(first_frame_mask_source, first_frame_mask_target)
        print(f"Copied {first_frame_mask_source} to {first_frame_mask_target}")
    else:
        print(f"Mask for the first frame {first_frame_mask_source} does not exist.")

    # Copy cam_K.txt to realworld
    if os.path.exists(cam_k_file_source):
        shutil.copy(cam_k_file_source, cam_k_file_target)
        print(f"Copied {cam_k_file_source} to {cam_k_file_target}")
    else:
        print(f"{cam_k_file_source} does not exist.")
    
    
    print('Step-2: Get real time rgb, depth, masks from robot')

    # image listener
    frames = 15
    curr_frame = 1
    input(f"Continue to get real time rgbd+gsam-mask on {frames} frames?")
    while not rospy.is_shutdown() and curr_frame <= frames:
        try:
            img, depth, mask = listener.run_network()

            # File paths
            rgb_path = os.path.join(realworld_rgb_dir, "{:06d}.jpg".format(curr_frame))
            depth_path = os.path.join(realworld_depth_dir, "{:06d}.png".format(curr_frame))
            mask_path = os.path.join(realworld_mask_dir, "{:06d}.png".format(curr_frame))

            cv2.imwrite(depth_path, depth)
            mask[mask > 0] = 255
            cv2.imwrite(mask_path, mask)
            
            PILImg.fromarray(img).save(rgb_path)
            # PILImg.fromarray(depth).save(depth_path, format='PNG')

            curr_frame += 1
        except:
            continue

    input("Continue to run bundlesdf?")

    input("Continue to plot bundlesdf output?")
    
    for tag in ["pre", "post"]:
        print(f"Continuing for {tag} base optimization: collect mask?")
        pre_post_mask_capture(listener)

    # Load images
    final_file = "{:06d}".format(frames)

    img1 = cv2.imread(f"{realworld_rgb_dir}/000000.jpg")
    img2 = cv2.imread(f"{realworld_rgb_dir}/{final_file}.jpg")

    # Load pose matrices
    pose1 = np.loadtxt(f"{realworld_dir}/bundlesdf/ob_in_cam/000000.txt")
    pose2 = np.loadtxt(f"{realworld_dir}/bundlesdf/ob_in_cam/{final_file}.txt")

    # Camera intrinsic matrix (example)
    K = np.loadtxt(cam_k_file_source)

    # Step 1: Draw the reference pose (pose1) and relative pose (pose2) in the first image
    relative_pose_1 = np.linalg.inv(pose1) @ pose2  # Pose2 relative to Pose1
    img1_with_pose = draw_pose_axis(img1.copy(), pose1, K)  # Draw pose1
    img1_with_pose = draw_pose_axis(img1_with_pose, relative_pose_1, K, color_scheme=[(255, 0, 255), (255, 255, 0), (0, 255, 255)])  # Draw pose2 relative to pose1

    # Step 2: Draw the reference pose (pose1) and relative pose (pose2) in the second image
    img2_with_pose = draw_pose_axis(img2.copy(), pose1, K)  # Draw pose1 in the second image
    img2_with_pose = draw_pose_axis(img2_with_pose, pose2, K)  # Draw pose2 directly

    # Save the output images
    cv2.imwrite("000000_with_pose.png", img1_with_pose)
    cv2.imwrite(f"{final_file}_with_pose.png", img2_with_pose)

    print(f"Poses drawn and saved as '000000_with_pose.png' and '{final_file}_with_pose.png'.")
