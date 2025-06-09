
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

- Following modules are tested on **Python 3.10.15**
  - robokit (gdino+samv2)
  - hamer
  - rfp-grasp-transfer
- BundleSDF runs in docker with **Python 3.8**
---

https://github.com/user-attachments/assets/015088f9-7031-44b9-b1b4-f4ea75043109

## 🔧 Tools

- Step-1 is needed for Step-2.
- Step-3 can be performed independently.
- Step-4 depends on the output of Step-3.

### 1. 🤖 Testing GDINO Prompts
To test GDINO with a text prompt:
```shell
cp scripts/test_gdino_prompts.py .
python test_gdino_prompts.py --input_dir ./imgs/test/000100/rgb --text_prompt <obj-text-prompt> --infer_first_only
# The output will be saved in: /imgs/test/000100/out/gdino/<obj_text_prompt>
```

Once you have a good text prompt that can detect object of interest, use it in step-2.

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
![vie-hand](../media/data_capture/vie-hand.png)

To extract right(1) / left(0) hand bounding boxes and 3D meshes
- Assuming only one person in the scene
- <red style="color:red">Frames containing atleast one hand will be only saved in `out/hamer/model`</red>
```shell
cd hamer
python extract_hand_bboxes_and_meshes.py --intrinsic_of umi_ft_fetch --opt_weight 100.0 --input_dir "../imgs/test/000100/rgb"

# Output will be saved in:
# ../imgs/test/000100/out/hamer/extra_plots - For visualization and debugging
# ../imgs/test/000100/out/hamer/scene - RGB scene point cloud
# ../imgs/test/000100/out/hamer/model - HAMER output, including mano params
# ../imgs/test/000100/out/hamer/3dhand - 3D hand mesh aligned with scene point cloud
```

- If you find the below error, it is likely due to Python3.10
```
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections' 
```

### 4. Human Hand to Fetch Gripper Transfer

This step can only be performed after getting the output from step-3 as it needs the human hand mesh.

```shell
# cd & git submodule update
cd rfp-grasp-transfer
git submodule update --init --recursive

# run transfer script
python transfer_from_hamer.py \
--mano_model_dir ../hamer/_DATA/data/mano/mano_v1_2/models/ \
--target_gripper fetch_gripper --debug_plots \
--input_dir ../_DATA/human_demonstrations/fetch-shelf-ycb-red-mug_interval_0.05/
```

If you encounter the following error:
```
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections'
```
try: `pip install --upgrade networkx`



### BundleSDF
```shell
cd BundleSDF/
python run_custom.py --mode run_video --video_dir ../data/bsdf-demo/ --out_folder ../data/bsdf-demo/out/bundlesdf --use_segmenter 0 --use_gui 1 --debug_level 2
```

python run_custom.py --mode run_video --video_dir ../realworld --out_folder ../realworld/out --use_segmenter 0 --use_gui 1

If you encounter the following error:
```
from ._ckdtree import cKDTree, cKDTreeNode
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/conda/envs/py38/lib/python3.8/site-packages/scipy/spatial/_ckdtree.cpython-38-x86_64-linux-gnu.so)
```
try: `pip install --upgrade scipy==1.10 yacs`

- If the pose region is within the mask provided then, we can omit it and move on to the next frame. This is one heuristic for eliminating incorrect poses.

## To run GSAM and BundleSDF together for realworld pose estimation
```shell
./run_obj_pose_est.sh "./vie/_DATA/new-data-from-fetch-and-laptop/22tasks.latest/task_8_17s-use_hammer/" "blue hammer" 15 5
```

### After data processing, following would be the dir structure
```
├── data_captured
    ├── <task-name>-1/
        ├── cam_K.txt
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
            └── bundlesdf
                ├── demonstration
                    ├── ob_in_cam
                    ├── 
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
- [GDINO + SamV2](https://github.com/jishnujayakumar/robokit)
- [HaMeR](https://github.com/IRVLUTD/HaMeR)
- [rfp-grasp-transfer](https://github.com/IRVLUTD/rfp-grasp-transfer)
- [BundleSDF](https://github.com/jishnujayakumar/BundleSDF)
---
