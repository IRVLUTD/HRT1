
# 📁 Project Setup and Usage Guide

## 🛠️ Setup Instructions

To set up the environment and prepare the project, run the following commands:

### 🧑‍💻 Run the Setup Script
```shell
# Remove all __pycache__ directories and .egg-info files recursively
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.egg-info" -type d -exec rm -rf {} +

# Make the setup script executable and run it
chmod +x ./setup_vie.sh
./setup_vie.sh
```

---

## 📜 Requirements

- All the vie modules are tested on **Python 3.10.15**
  - robokit
  - gdino
  - samv2
  - hamer

---

https://github.com/user-attachments/assets/015088f9-7031-44b9-b1b4-f4ea75043109

## 🔧 Tools

### 1. 🤖 Testing GDINO Prompts
To test GDINO with a text prompt:
```shell
cp scripts/test_gdino_prompts.py .
python test_gdino_prompts.py --input_dir ./imgs/test/000100/rgb --text_prompt <obj-text-prompt> --infer_first_only
# The output will be saved in: /imgs/test/000100/out/gdino/<obj_text_prompt>
```

### 2. 🔍 Testing GDINO + SAMv2
To use GDINO and SAMv2 for object bounding box detection and tracking in video frames:
```shell
cp scripts/test_gdino_samv2.py .
python test_gdino_samv2.py --input_dir ./imgs/test/000100/rgb --text_prompt <obj-text-prompt> --save_interval=1
# Output saved in:
# ../imgs/test/000100/out/samv2/<obj_text_prompt>/obj_masks - object mask
# ../imgs/test/000100/out/samv2/<obj_text_prompt>/masks_traj_overlayed - Trajectory + mask overlay + initial object bbox
```

### 3. ✋ Extracting Right/Left Hand BBoxes and Meshes
![vie-hand](../media/imgs/vie-hand.png)

To extract right(1) / left(0) hand bounding boxes and 3D meshes
- Assuming only one person in the scene
- <red style="color:red">Frames containing atleast one hand will be only saved in `out/hamer/model`</red>
```shell
cd hamer
python extract_hand_bboxes_and_meshes.py --input_dir "../imgs/test/000100/rgb"

# Output will be saved in:
# ../imgs/test/000100/out/hamer/extra_plots - For visualization and debugging
# ../imgs/test/000100/out/hamer/scene - RGB scene point cloud
# ../imgs/test/000100/out/hamer/model - HAMER output, including mano params
# ../imgs/test/000100/out/hamer/3dhand - 3D hand mesh aligned with scene point cloud
```

### After data processing, following would be the dir structure
```
├── data_captured
    ├── <task-name>-1/
        ├── rgb/
            ├── 000000.jpg
            ├── 000001.jpg
            └── ...
        ├── depth/
            ├── 000000.png
            ├── 000001.png
            └── ...
        ├── pose/
            ├── 000000.npz
            ├── 000001.npz
            └── ...
        └── out/
            ├── gdino
                ├── <text-prompt>
            ├── samv2
                ├── <text-prompt>
                ├── obj_masks
                └── masks_traj_overlayed
            └── hamer
                ├── extra_plots
                    ├── 000000.npz
                    ├── 000000.npz
                    ├── 000000.npz
                ├── scene
                    ├── 000000.ply
                    ├── 000001.ply
                    └── ...
                ├── model
                    ├── 000000.npz
                    ├── 000001.npz
                    └── ...
                └── 3dhand
                    ├── 000000.ply
                    ├── 000001.ply
                    └── ...
        
    ├── <task-name>-2/
    └── <task-name>-.../
```

---

## 🙏 Acknowledgments

This project utilizes the following resources:

- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)
---
