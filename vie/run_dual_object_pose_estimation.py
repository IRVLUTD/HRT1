#!/usr/bin/env python
import os
import sys
import cv2
import time  # Added this import
import rospy
import argparse
import numpy as np
import yaml as pyyaml
from PIL import Image as PILImg
from sensor_msgs.msg import Image as RosImage
from ros_listener import ImageListener
from bundlesdf_processor import BundleSDFProcessor
from utils import read_obj_prompts, prettify_prompt, create_symlink, remove_symlink_only, copy_file_if_exists, create_required_out_folders

def make_yaml_safe(data):
    """Recursively convert NumPy types and other non-YAML-safe types to Python built-in types."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.floating, np.complexfloating)):
        return float(data)
    elif isinstance(data, (np.integer, np.bool_)):
        return int(data)
    elif isinstance(data, dict):
        return {make_yaml_safe(k): make_yaml_safe(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_yaml_safe(item) for item in data]
    elif isinstance(data, set):
        return [make_yaml_safe(item) for item in sorted(data)]
    elif data is None or isinstance(data, (str, int, float, bool)):
        return data
    else:
        # Handle other types by converting to string as a fallback
        return str(data)

def print_object_count(objx, prompt, mask):
    """
    Print formatted output with object count, colored red if > 1, green otherwise.
    
    Args:
        objx: Object identifier (e.g., index or name)
        prompt: Text prompt to be prettified
        mask: Numpy array containing object mask
        np: Numpy module
    """
    num_objects = len(np.unique(mask)) - 1
    color_code = "\033[31m" if num_objects > 1 else "\033[32m"
    reset_code = "\033[0m"
    colored_num = f"{color_code}{num_objects}{reset_code}"
    print("{:<10} | {:<20} | {:>15}".format(objx, prettify_prompt(prompt), colored_num))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="run_video", help="run_video/global_refine/draw_pose")
    parser.add_argument("--task_dir", type=str, required=True, help="Directory containing task data")
    parser.add_argument("--video_dir", type=str, default="realworld", help="Directory for video data")
    parser.add_argument("--out_folder", type=str, default="realworld/out/bundlesdf/rollout", help="Output folder")
    parser.add_argument("--use_segmenter", type=int, default=0, help="Use segmenter (0 or 1)")
    parser.add_argument("--use_gui", type=int, default=0, help="Use GUI (0 or 1)")
    parser.add_argument("--stride", type=int, default=1, help="Frame interval")
    parser.add_argument("--debug_level", type=int, default=0, help="Debug level")
    parser.add_argument("--demo_frames", type=int, default=15, help="Number of frames to collect")
    parser.add_argument("--rollout_frames", type=int, default=5, help="Number of frames to collect")
    parser.add_argument("--camera", type=str, default="Fetch", help="Camera type (Fetch, Realsense, Azure, etc.)")
    args = parser.parse_args()

    rospy.init_node("merged_bundlesdf_node", anonymous=True)
    pub_row0_rgb = rospy.Publisher("/bundleSDF/src_rgb_with_pose", RosImage, queue_size=10)
    pub_row0_masked_rgb = rospy.Publisher("/bundleSDF/src_masked_rgb", RosImage, queue_size=10)
    pub_row1_rgb = rospy.Publisher("/bundleSDF/dst_rgb_with_pose", RosImage, queue_size=10)
    pub_row1_masked_rgb = rospy.Publisher("/bundleSDF/dst_masked_rgb", RosImage, queue_size=10)

    root_dir = os.path.normpath(args.task_dir)
    realworld_dir = args.video_dir
    rgb_source_dir = os.path.join(root_dir, "rgb")
    depth_source_dir = os.path.join(root_dir, "depth")
    pose_source_dir = os.path.join(root_dir, "pose")
    sam2_out_dir = os.path.join(root_dir, "out", "samv2")
    bundle_sdf_out_dir = os.path.join(root_dir, "out", "bundlesdf")
    bsdf_demo_dir = os.path.join(bundle_sdf_out_dir, "demonstration")

    obj_prompt_mapper = read_obj_prompts(bsdf_demo_dir)
    listener = ImageListener(camera=args.camera)
    rospy.sleep(0.1)

    timing_results = {}

    for objx, prompt in obj_prompt_mapper.items():
        try:
            mask_source_dir = os.path.join(sam2_out_dir, prompt, "obj_masks")
        except IndexError:
            print(f"Error: No SAMv2 output directories found in {sam2_out_dir}")
            sys.exit(1)
        realworld_rgb_dir = os.path.join(realworld_dir, "rgb")
        realworld_depth_dir = os.path.join(realworld_dir, "depth")
        realworld_pose_dir = os.path.join(realworld_dir, "pose")
        realworld_mask_dir = os.path.join(realworld_dir, f"masks_{objx}")
        realworld_bsdf_demo_dir = os.path.join(realworld_dir, "out/bundlesdf/demonstration", objx)
        realworld_bsdf_oic_dir = os.path.join(realworld_bsdf_demo_dir, "ob_in_cam")
        realworld_bsdf_pose_overlay_dir = os.path.join(realworld_bsdf_demo_dir, "pose_overlayed_rgb")
        create_required_out_folders(realworld_bsdf_demo_dir)
        cam_k_file_source = os.path.join(root_dir, "cam_K.txt")
        cam_k_file_target = os.path.join(realworld_dir, "cam_K.txt")
        obj_prompt_mapper_src = os.path.join(bsdf_demo_dir, "obj_prompt_mapper.json")
        obj_prompt_mapper_dst = os.path.join(os.path.dirname(realworld_bsdf_demo_dir), "obj_prompt_mapper.json")
        copy_file_if_exists(cam_k_file_source, cam_k_file_target)
        copy_file_if_exists(obj_prompt_mapper_src, obj_prompt_mapper_dst)

        rospy.loginfo(f"Collecting {args.demo_frames} frames from source directories...")
        for i in range(args.demo_frames):
            frame_str = f"{i:06d}"
            files = [
                (rgb_source_dir, realworld_rgb_dir, f"{frame_str}.jpg", "RGB"),
                (depth_source_dir, realworld_depth_dir, f"{frame_str}.png", "Depth"),
                (mask_source_dir, realworld_mask_dir, f"{frame_str}.png", "Mask"),
                (pose_source_dir, realworld_pose_dir, f"{frame_str}.npz", "Pose"),
                (os.path.join(bsdf_demo_dir, objx, "ob_in_cam"), realworld_bsdf_oic_dir, f"{frame_str}.txt", "obj in camera pose txt"),
                (os.path.join(bsdf_demo_dir, objx, "pose_overlayed_rgb"), realworld_bsdf_pose_overlay_dir, f"{frame_str}.png", "BSDF Pose Overlayed RGB"),
            ]
            for src_dir, tgt_dir, filename, label in files:
                copy_file_if_exists(os.path.join(src_dir, filename), os.path.join(tgt_dir, filename))

    print("Step-2: Get real-time RGB, depth, masks from robot")
    curr_frame = args.demo_frames
    print(f"Collecting real-time RGBD+SAM-mask for {args.rollout_frames} frame(s)...")
    time.sleep(0.1)
    timing_results["mask_prediction"] = {}

    print("\n{:<10} | {:<20} | {:>15}".format("Object ID", "Text Prompt", "Objects Detected"))
    print("-" * 10 + "|" + "-" * 21 + "|" + "-" * 16)
    
    obj_prompt_mapper_rollout = obj_prompt_mapper.copy()
    im_color, depth_img, rgb_frame_id, rgb_frame_stamp, RT_camera, RT_laser = listener.get_latest_listener_data()

    for objx, prompt in obj_prompt_mapper_rollout.items():
        while True:
            # Generate the mask and annotated image using the current prompt
            bboxes, phrases, gdino_conf = listener.get_gdino_preds(im_color, prettify_prompt(prompt))
            
            print(bboxes, phrases)

            # Ask the user if the current prompt is correct
            user_input = input(f"Current text prompt is '{prompt}'. Is this correct? (Y/N): ").strip().lower()
            
            # Validate user input
            while user_input not in ['y', 'n']:
                print("Invalid input. Please enter 'Y' for yes or 'N' for no.")
                user_input = input(f"Current text prompt is '{prompt}'. Is this correct? (Y/N): ").strip().lower()
            
            if user_input == 'y':
                # If the prompt is correct, save it and move on to the next object
                obj_prompt_mapper_rollout[objx] = prompt
                break
            else:
                # If the prompt is not correct, ask for a new prompt
                prompt = input("Please enter your custom prompt: ").strip()
                obj_prompt_mapper_rollout[objx] = prompt


    while (
        not rospy.is_shutdown() and curr_frame <= args.demo_frames + args.rollout_frames
    ):
        try:
            im_color, depth_img, rgb_frame_id, rgb_frame_stamp, RT_camera, RT_laser = listener.get_latest_listener_data()
            if im_color is None:
                continue
            # im_color = im_color[:, :, (2, 1, 0)]  # BGR to RGB
            for objx, prompt in obj_prompt_mapper_rollout.items():
                bbox_annotated_img, mask, mask_time = listener.get_gsam_mask(im_color, prettify_prompt(prompt))
                listener.publish_overlay(bbox_annotated_img, rgb_frame_stamp, rgb_frame_id)
                listener.publish_mask(mask, rgb_frame_stamp, rgb_frame_id)
                print_object_count(objx, prettify_prompt(prompt), mask)
                realworld_mask_dir = os.path.join(realworld_dir, f"masks_{objx}")
                rgb_path = os.path.join(realworld_rgb_dir, f"{curr_frame:06d}.jpg")
                depth_path = os.path.join(realworld_depth_dir, f"{curr_frame:06d}.png")
                mask_path = os.path.join(realworld_mask_dir, f"{curr_frame:06d}.png")
                pose_path = os.path.join(realworld_pose_dir, f"{curr_frame:06d}.npz")
                cv2.imwrite(depth_path, depth_img)
                mask[mask > 0] = 255
                cv2.imwrite(mask_path, mask)
                PILImg.fromarray(im_color).save(rgb_path)
                np.savez(pose_path, RT_camera=RT_camera)
                timing_results["mask_prediction"].setdefault(objx, []).append({"frame": curr_frame, "time": mask_time})
                timing_results[f"{objx}_text_prompt"] = prompt
            print("-" * 11 + "*" + "-" * 22 + "*" + "-" * 16)
            curr_frame += 1
        except Exception as e:
            print(f"Error in frame {curr_frame}: {e}")
            continue

    print("Running BundleSDF...")
    K = listener.intrinsics
    masks_symlink_path = os.path.join(realworld_dir, "masks")
    rollout_details = {}
    timing_results["bundlesdf_processing"] = {}

    for objx, prompt in obj_prompt_mapper.items():
        realworld_mask_dir = os.path.abspath(os.path.join(realworld_dir, f"masks_{objx}"))
        create_symlink(realworld_mask_dir, masks_symlink_path)
        print(f"Created symlink: {masks_symlink_path} -> {realworld_mask_dir}")
        processor = BundleSDFProcessor(
            video_dir=args.video_dir,
            out_folder=f"{args.out_folder}/{objx}",
            use_segmenter=args.use_segmenter,
            use_gui=args.use_gui,
            stride=args.stride,
            debug_level=args.debug_level,
            K=K,
        )
        try:
            process_time = processor.process(pub_row0_rgb, pub_row0_masked_rgb, pub_row1_rgb, pub_row1_masked_rgb)
            rollout_details[objx] = {
                "out_folder": processor.out_folder,
                "obj_prompt": prettify_prompt(prompt),
            }
            timing_results["bundlesdf_processing"][objx] = process_time
        except Exception as e:
            print(f"Error in BundleSDF processing for {objx}: {e}")
        finally:
            remove_symlink_only(masks_symlink_path)
            print(f"Removed symlink: {masks_symlink_path}")

    rollout_details["K"] = {
        "rollout": {},
        "demonstration": {}
    }

    rollout_details["K"]["rollout"]["fx"] = float(K[0,0])
    rollout_details["K"]["rollout"]["fy"] = float(K[1,1])
    rollout_details["K"]["rollout"]["cx"] = float(K[0,2])
    rollout_details["K"]["rollout"]["cy"] = float(K[1,2])
    rollout_details["args"] = vars(args)
    rollout_details["timing_results"] = timing_results

    cam_K = np.loadtxt(cam_k_file_source)
    rollout_details["K"]["demonstration"]["fx"] = float(cam_K[0,0])
    rollout_details["K"]["demonstration"]["fy"] = float(cam_K[1,1])
    rollout_details["K"]["demonstration"]["cx"] = float(cam_K[0,2])
    rollout_details["K"]["demonstration"]["cy"] = float(cam_K[1,2])

    print("\nTiming Results:")
    for objx in timing_results["mask_prediction"]:
        times = [t["time"] for t in timing_results["mask_prediction"][objx]]
        avg_time = np.mean(times) if times else 0
        print(f"Object {objx} Mask Prediction - Avg Time: {avg_time:.3f} seconds, Frames: {len(times)}")
        rollout_details[objx]["mask_prediction_time"] = avg_time
    for objx, proc_time in timing_results["bundlesdf_processing"].items():
        print(f"Object {objx} BundleSDF Processing - Total Time: {proc_time:.3f} seconds")
        rollout_details[objx]["bundlesdf_processing_time"] = proc_time

    # Convert rollout_details to YAML-safe types
    rollout_details_safe = make_yaml_safe(rollout_details)
    rollout_details_path = os.path.join(realworld_dir, "rollout_details.yaml")

    with open(rollout_details_path, "w") as f:
        pyyaml.safe_dump(rollout_details_safe, f, sort_keys=False)


if __name__ == "__main__":
    main()