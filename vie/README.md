
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

## 📜 Requirements

- Following modules are tested on **Python 3.10.15**
  - robokit (gdino+samv2)
  - hamer
  - rfp-grasp-transfer
- BundleSDF runs in [docker](BundleSDF/docker) with **Python 3.8**

https://github.com/user-attachments/assets/015088f9-7031-44b9-b1b4-f4ea75043109


## Data preprocessing

### 1. Setup paths
```shell
export PROJECT_ROOT=/path/to/hrt1
export VIE_ROOT=$PROJECT_ROOT/vie
export DATA_ROOT=/path/to/data/captured # data_captured/task_x
```

## 🔧 Tools

- Step-2 is needed for Step-3.
- Step-3 is needed for Step-6.
- Step-4 can be performed independently.
- Step-5 depends on the output of Step-3.
- Step-7 is for realworld execution.

---

### 2. 🔍 Testing GDINO Prompts
First detect the object of interest in the first frame using GDINO with a text prompt:
```shell
cd $VIE_ROOT
python run_gdino_samv2.py --input_dir $DATA_ROOT/rgb --text_prompt <obj-text-prompt> --infer_first_only
# The output will be saved in: $DATA_ROOT/out/gdino/<obj_text_prompt>
# if text prompt contains space " " then it will be replaced by "_"
```
Once you have a good text prompt that can detect object of interest, use it in step-3.

---

### 3. 🤖 Generate object masks using GDINO + SAMv2
To use GDINO and SAMv2 for object bounding box detection and tracking in video frames:
```shell
cd $VIE_ROOT
python run_gdino_samv2.py --input_dir $DATA_ROOT/rgb --text_prompt <obj-text-prompt> --save_interval=1
# Output saved in:
# $DATA_ROOT/out/samv2/<obj_text_prompt>/obj_masks - object mask
# $DATA_ROOT/out/samv2/<obj_text_prompt>/masks_traj_overlayed - Trajectory + mask overlay + initial object bbox
```

---

### 4. ✋ Extracting Right/Left Hand BBoxes and 3D Meshes (HaMeR)
![vie-hand](../media/data_capture/vie-hand.png)
This step extracts right(1) / left(0) hand bounding boxes and 3D hand meshes using HaMeR.

✅ Assumptions:
- Only one person is present in the scene.
- Only frames containing at least one visible hand will be processed and saved under `out/hamer/model`.
```shell
cd $VIE_ROOT/hamer
python extract_hand_bboxes_and_meshes.py \
--intrinsic_of umi_ft_fetch \
--opt_weight 100.0 \
--input_dir $DATA_ROOT/rgb
```

📤 Output Directory Structure:
- $DATA_ROOT/out/hamer/extra_plots – Visualizations and debugging images
- $DATA_ROOT/out/hamer/scene – RGB scene point cloud
- $DATA_ROOT/out/hamer/model – HaMeR results including MANO parameters
- $DATA_ROOT/out/hamer/3dhand – Aligned 3D hand meshes

🛠️ Known Issue (Python 3.10+)
If you encounter:
```shell
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections'
```
✅ Fix: `pip install --upgrade networkx`

---

### 5. Transfer Human Hand to Fetch Gripper

This step requires the human hand mesh output from Step 4 (HaMeR). It maps the human hand configuration to the target robot gripper (e.g., Fetch gripper).

```shell
# Navigate to the repo and initialize submodules
cd $VIE_ROOT/rfp-grasp-transfer
git submodule update --init --recursive

# Run the hand-to-gripper transfer script
python transfer_from_hamer.py \
--mano_model_dir ../hamer/_DATA/data/mano/mano_v1_2/models/ \
--target_gripper fetch_gripper \
--debug_plots \
--input_dir $DATA_ROOT
```

📤 Output Directory Structure:
- $DATA_ROOT/out/hamer/transfer_extra_plots – Visualizations and debugging plots
- $DATA_ROOT/out/hamer/transfer_hand_mesh – Transfered 3D fetch gripper meshes

🛠️ Troubleshooting
If you see this error:
```
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections'
```
✅ Try this fix: `pip install --upgrade networkx`


### 6. Object Pose Estimation Using BundleSDF
Run object pose estimation on captured video frames using BundleSDF:
```shell
cd $VIE_ROOT/BundleSDF/
python run_pose_only_bsdf.py --mode run_video --video_dir $DATA_ROOT
```

🛠️ Troubleshooting
If you encounter the following error:
```shell
from ._ckdtree import cKDTree, cKDTreeNode
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/conda/envs/py38/lib/python3.8/site-packages/scipy/spatial/_ckdtree.cpython-38-x86_64-linux-gnu.so)
```
✅ Try this fix: `pip install --upgrade scipy==1.10 yacs`

💡 Heuristic Tip:
- If the predicted pose lies entirely within the object mask, it can be considered valid.
- If not, the frame can be skipped—this simple heuristic helps filter out incorrect poses efficiently.

---

## 7. Run GSAM2 + BundleSDF for Real-World Object Pose Estimation
Use the following command to perform real-time object pose estimation by combining source frames (human demo) with rollout frames (robot execution):
```shell
cd $VIE_ROOT
./run_obj_pose_est.sh $DATA_ROOT <text-prompt> <src-frames> <rollout-frames>
# Example:
# ./run_obj_pose_est.sh "./vie/_DATA/new-data-from-fetch-and-laptop/22tasks.latest/task_8_17s-use_hammer/" "blue hammer" 15 5
```
🧩 Arguments:
- `text-prompt`: The object name or description used for GDINO+SAMv2 (e.g., "blue hammer").
- `src-frames`: Number of frames to extract from the human demonstration.
- `rollout-frames`: Number of frames to process during real-time rollout.

## 👉 Object Pose Estimation with Multi-Frame Context:
- 📸 Uses multiple frames from both:
  - Source phase (human demonstration)
  - Rollout phase (real-time robot execution)
- 🎯 Provides richer visual and temporal context for improved understanding
- 🚀 Significantly outperforms single-frame methods in:
  - Accuracy
  - Stability
  - Robustness
- ✅ Enables more reliable pose tracking across time during real-world execution

## 🗂️ Output Directory Structure After Data Processing
```
data_captured/
├── task_1/
│   ├── cam_K.txt                      # Camera intrinsics
│   ├── rgb/                           # RGB frames
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── depth/                         # Aligned depth frames (in mm)
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── pose/                          # RT camera (npz)
│   │   ├── 000000.npz
│   │   ├── 000001.npz
│   │   └── ...
│   └── out/                           # All derived outputs
│       ├── gdino/
│       │   └── <text-prompt>/         # GDINO raw detection outputs
│       ├── samv2/
│       │   ├── <text-prompt>/             # Processed masks from GDINO + SAMv2
    │       │   ├── obj_masks/             # Binary masks per object
    │       │   └── masks_traj_overlayed/  # RGB overlays with tracked masks
│       ├── bundlesdf/
│       │   ├── demonstration/
│       │   │   └── obj_<1/2>/
│       │   │       ├── ob_in_cam/             # Object point clouds
│       │   │       ├── pose_overlayed_rgb/    # Pose visualizations
│       │   │       └── obj_prompt_mapper.json # Maps object index to prompt
│       │   └── rollout/
│       │       └── obj_<1/2>/                 # Same as demo but for execution
│       │           ├── ob_in_cam/
│       │           ├── pose_overlayed_rgb/
│       │           └── obj_prompt_mapper.json
│       └── hamer/
│           ├── extra_plots/          # Debug plots (optional)
│           │   ├── 000000.npz
│           │   └── ...
│           ├── scene/                # Full scene point clouds
│           │   ├── 000000.ply
│           │   └── ...
│           ├── model/                # MANO hand model outputs
│           │   ├── 000000.npz
│           │   └── ...
│           └── 3dhand/               # 3D hand meshes aligned to scene
│               ├── 000000.ply
│               └── ...
├── task_2/
└── task_.../
```


### 🗂️ obj_prompt_mapper.json
- Maps object identifiers (e.g., obj_1, obj_2) to their corresponding text prompts used during SAMv2 mask generation.
- Ensures a consistent mapping between:
  - Source: $DATA_ROOT/out/sam2/<text_prompt>/obj_masks
  - Target: $DATA_ROOT/masks/
- This linkage is critical for enabling accurate object pose estimation during real-time execution.

🔗 Example:
```json
{
  "obj_1": "bin",
  "obj_2": "sonic"
}
```


## 🙏 Acknowledgments

This project utilizes the following resources:
- [GDINO + SamV2](https://github.com/jishnujayakumar/robokit)
- [HaMeR](https://github.com/IRVLUTD/HaMeR)
- [rfp-grasp-transfer](https://github.com/IRVLUTD/rfp-grasp-transfer)
- [BundleSDF](https://github.com/jishnujayakumar/BundleSDF)
