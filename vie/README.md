
# 📁 Project Setup and Usage Guide

## 🛠️ Setup Instructions

To set up the environment and prepare the project, run the following commands:

### 🧑‍💻 Run the Setup Script
```shell
# Remove all __pycache__ directories and .egg-info files recursively
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.egg-info" -type d -exec rm -rf {} +

# Make the setup script executable and run it
chmod +x ./setup_perception.sh
./setup_perception.sh
```

---

## 📜 Requirements

- All the vie modules are tested on **Python 3.10.15**
  - robokit
  - gdino
  - samv2
  - hamer

---

## 🔧 Tools

### 1. 👁️ Viewing `.obj` Files
To visualize `.obj` files, use the following script:
```shell
python scripts/vis_obj.py <path/to/.obj>
```

### 2. 🤖 Testing GDINO Prompts
To test GDINO with a text prompt:
```shell
python test_gdino_prompts.py --input_dir ./imgs/irvl-whiteboard-write-and-erase/rgb --text_prompt "black eraser" --infer_first_only
# The output will be saved in: ./imgs/gdino/irvl-whiteboard-write-and-erase/black_eraser
```

### 3. 🔍 Testing GDINO + SAMv2
To use GDINO and SAMv2 for object bounding box detection and tracking in video frames:
```shell
python test_gdino_samv2.py --input_dir ./imgs/irvl-whiteboard-write-and-erase/rgb --text_prompt "black eraser" --save_interval=1
# Output saved in:
# ./imgs/irvl-whiteboard-write-and-erase/samv2/black_eraser/obj_masks - object mask
# ./imgs/irvl-whiteboard-write-and-erase/samv2/black_eraser/masks_traj_overlayed - Trajectory + mask overlay + initial object bbox
```

### 4. 🖥️ Testing HAMER
To test the HAMER pipeline for processing images:
```shell
cd hamer
python demo.py --img_folder ../imgs/irvl-whiteboard-write-and-erase/rgb/ --out_folder irvl-whiteboard-write-and-erase-test --batch_size=48 --side_view --save_mesh --full_frame
```

### 5. ✋ Extracting Right/Left Hand BBoxes and Meshes
To extract right(1)/left(0) hand bounding boxes and 3D meshes (assuming only one person in the scene):
```shell
cd hamer
python extract_hand_bboxes_and_meshes.py --input_dir "../imgs/irvl-whiteboard-write-and-erase/rgb/"
# Output will be saved in:
# ./imgs/irvl-whiteboard-write-and-erase/hamer/extra_plots  - For visualization and debugging
# ./imgs/irvl-whiteboard-write-and-erase/hamer/scene  - RGB scene point cloud
# ./imgs/irvl-whiteboard-write-and-erase/hamer/model  - HAMER output, including mano params
# ./imgs/irvl-whiteboard-write-and-erase/hamer/3dhand  - 3D hand mesh aligned with scene point cloud
```

---

## 🙏 Acknowledgments

This project utilizes the following resources:

- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)

---
