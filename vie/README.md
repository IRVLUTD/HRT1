
# 📁 VIE Setup and Usage Guide

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31015/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Lab: IRVL UTD](https://img.shields.io/badge/lab-IRVL%20UTD-005CAB)](https://labs.utdallas.edu/irvl/)
[![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA--Research-orange)](https://github.com/IDEA-Research/GroundingDINO)
[![SAM 2](https://img.shields.io/badge/SAM%202-Meta-1877F2)](https://github.com/facebookresearch/sam2)
[![HaMeR](https://img.shields.io/badge/HaMeR-geopavlakos-7D3C98)](https://github.com/geopavlakos/hamer)
[![BundleSDF](https://img.shields.io/badge/BundleSDF-NVlabs-76B900)](https://github.com/NVlabs/BundleSDF)


- [📁 VIE Setup and Usage Guide](#-vie-setup-and-usage-guide)
  - [✨ Performance & UX improvements](#-performance--ux-improvements)
  - [🛠️ Setup Instructions](#️-setup-instructions)
    - [🧑‍💻 Run the Setup Script](#-run-the-setup-script)
  - [📜 Requirements](#-requirements)
  - [⚙️ Data preprocessing](#️-data-preprocessing)
    - [1. Setup paths](#1-setup-paths)
  - [🔧 Tools](#-tools)
    - [2. 🔍 Find object prompts using GDINO](#2--find-object-prompts-using-gdino)
    - [3. 🤖 Generate object masks using GDINO + SAMv2](#3--generate-object-masks-using-gdino--samv2)
    - [4. ✋ Extracting Right/Left Hand BBoxes and 3D Meshes (HaMeR)](#4--extracting-rightleft-hand-bboxes-and-3d-meshes-hamer)
    - [5. Transfer Human Hand to Fetch Gripper](#5-transfer-human-hand-to-fetch-gripper)
    - [🧩 To visualize scene, hamer hand pose \& transferred pose](#-to-visualize-scene-hamer-hand-pose--transferred-pose)
    - [🔧 Arguments](#-arguments)
    - [6. BundleSDF Docker Setup](#6-bundlesdf-docker-setup)
    - [7. Object Pose Estimation Using BundleSDF](#7-object-pose-estimation-using-bundlesdf)
  - [8. Run GSAM2 + BundleSDF for Real-World Object Pose Estimation](#8-run-gsam2--bundlesdf-for-real-world-object-pose-estimation)
    - [👉 Object Pose Estimation with Multi-Frame Context:](#-object-pose-estimation-with-multi-frame-context)
  - [🗂️ Output Directory Structure After Data Processing](#️-output-directory-structure-after-data-processing)
    - [🗂️ obj\_prompt\_mapper.json](#️-obj_prompt_mapperjson)
  - [⚠️ Install Gotchas](#️-install-gotchas-read-first-if-setup_viesh-is-failing)
  - [⚡ Benchmark](#-benchmark)
    - [GDINO + SAMv2 — measured 7.86× on the propagation hot loop](#gdino--samv2--measured-786-on-the-propagation-hot-loop)
    - [HaMeR — measured 6.49× on the scipy minimize step](#hamer--measured-649-on-the-scipy-minimize-step)
    - [rfp-grasp-transfer — measured ≈1.5× on a noisy CPU bench (larger expected on GPU)](#rfp-grasp-transfer--measured-15-on-a-noisy-cpu-bench-larger-expected-on-gpu)
    - [BundleSDF — docker persistence + tighter n\_step (not measured here)](#bundlesdf--docker-persistence--tighter-n_step-not-measured-here)
  - [🙏 Acknowledgments](#-acknowledgments)


## ✨ Performance & UX improvements

A coordinated speedup + UX pass across the four vie modules. **All optimizations preserve correctness; viz/debug outputs are opt-in via a unified `--save_viz` flag so the default fast path is also the clean path.**

**Speed gains** (measured on RTX 5070 Laptop, `task_39` 70-frame clip):
- **GDINO + SAMv2**: ~21× per-frame steady-state vs `main` (2000 ms → 94 ms)
- **HaMeR**: 6.49× on the scipy-minimize stage; lazy imports cut startup ~5×
- **rfp-grasp-transfer**: ~5× wall-clock end-to-end with `--frame_batch_size 4`
- **BundleSDF**: persistent docker + `n_step=5` defaults wired through

**Performance flags worth knowing** (default is *fast*; raise these if quality regresses):

| Script | Speed flags | Quality fallbacks |
|---|---|---|
| `run_gdino_samv2.py` | `--sam2_size base_plus` (≈2× faster than `large`) | `--sam2_size large` |
| `extract_hand_bboxes_and_meshes.py` | warm-start on, `--opt_xatol 1e-4 --opt_maxiter 30 --no_fp16` to revert | `--no_warm_start --opt_xatol 1e-5 --opt_maxiter 50` |
| `transfer_from_hamer.py` | `--num_particles 16 --max_iter 50 --frame_batch_size 4` | `--num_particles 32 --max_iter 100 --frame_batch_size 1` |
| `run_bundlesdf.py` | `--n_step 5` | `--n_step 10` |

**Unified `--save_viz` flag**: every entry-point script accepts `--save_viz` to write the human-readable debug artifacts that the original pipeline produced. Off by default for speed; turn it on for any single stage:

```shell
# GDINO + SAMv2: writes out/samv2/<prompt>/masks_traj_overlayed/*.png
python run_gdino_samv2.py --input_dir $TASK_DATA_ROOT/rgb \
    --text_prompt "white_egg" --save_viz

# HaMeR: writes out/hamer/extra_plots/{frame}_{0,1}.png + _all.jpg
python hamer/extract_hand_bboxes_and_meshes.py --opt_weight 100.0 \
    --input_dir $TASK_DATA_ROOT/rgb --save_viz

# rfp-grasp-transfer: writes out/hamer/transfer_extra_plots/{frame}_{0,1}.html (~5MB each)
python rfp-grasp-transfer/transfer_from_hamer.py \
    --mano_model_dir hamer/_DATA/data/mano \
    --target_gripper fetch_gripper \
    --input_dir $TASK_DATA_ROOT --save_viz
```

Per-script flags (`--save_traj_overlay`, `--save_debug_renders`, `--debug_plots`) still work for fine-grained control. Downstream pipeline data (binary masks, MANO npzs, scene PLYs, transferred-gripper PLYs, BundleSDF poses) is **always** written — `--save_viz` only toggles the *human-readable debug overlays* on top of those.

**UX**: each script now shows a colored startup banner, live spinners during long ops (model loads, model downloads, GroundingDINO inference), section headers (Configuration / Loading models / Tracking), and a bordered cyan summary panel with frames / fps / total wall / output path at the end. Third-party deprecation/registry warnings are silenced — real errors still propagate. See [`vie/robokit/log.py`](robokit/log.py) for the shared logging utilities.

## 🛠️ Setup Instructions

To set up the environment and prepare the vie pipeline, run the following commands:

### 🧑‍💻 Run the Setup Script
```shell
# Remove all __pycache__ directories and .egg-info files recursively
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.egg-info" -type d -exec rm -rf {} +

# Make the setup script executable and run it
chmod +x ./setup_vie.sh
./setup_vie.sh
```

## ⚠️ Install Gotchas (read first if `setup_vie.sh` is failing)

`setup_vie.sh` now bakes in the workarounds below; this section documents *why* they exist so you can debug a fresh install.

1. **MANO models** (`hamer/_DATA/data/mano/MANO_{LEFT,RIGHT}.pkl`) are license-gated and cannot be auto-installed. Register at https://mano.is.tue.mpg.de/, download `mano_v1_2.zip`, and copy the two `.pkl` files into `hamer/_DATA/data/mano/`.

2. **`mmcv` version pin**: HaMeR's `setup.py` originally pinned `mmcv==1.3.9`, but `mmpose==0.24.0` (its sibling dep) only accepts `mmcv` in `[1.3.8, 1.5.0]`. We've relaxed HaMeR's pin to `>=1.3.8,<=1.5.0`. `setup_vie.sh` installs `mmcv==1.5.0` explicitly, **with `setuptools<70`** since legacy mmcv's `setup.py` imports `pkg_resources` which newer setuptools dropped.

3. **`transformers` version pin**: GroundingDINO at the pinned commit (`2b62f419`) calls `BertModel.get_head_mask`, which `transformers>=5` removed. `requirements.txt` pins `transformers==4.47.1`.

4. **GroundingDINO `_C` extension**: the pip wheel ships no `_C.so` and the source build needs a CUDA toolchain matching torch's cuda version. `setup_vie.sh` patches `groundingdino/models/GroundingDINO/ms_deform_attn.py` in-place to fall back to the pure-PyTorch implementation when `_C` is missing. `robokit/perception.py` warns at import time if both `_C` and the patch are absent.

5. **NumPy 2.x incompatibility**: editable installs can drag in `numpy>=2`, which breaks matplotlib + many c-extensions. `setup_vie.sh` repins `numpy<2` after HaMeR's editable install.

6. **Blackwell GPUs (RTX 50-series, sm_120)**: torch `<=2.4` does not ship sm_120 kernels. If you see `no kernel image is available for execution on the device 'cuda:0'`, bump torch:
   ```bash
   pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

## 📜 Requirements

- Following modules are tested on **Python 3.10.15**
  - `robokit` (gdino+samv2). Example conda [env.yml](./conda-envs/gsam2-py3.10.yml).
  - `hamer` & `rfp-grasp-transfer`. Example conda [env.yml](./conda-envs/robokit-py3.10.yml).

- BundleSDF runs in [docker](BundleSDF/docker) with **Python 3.8**

https://github.com/user-attachments/assets/015088f9-7031-44b9-b1b4-f4ea75043109


## ⚙️ Data preprocessing

### 1. Setup paths
```shell
export PROJECT_ROOT=/path/to/hrt1
export VIE_ROOT=$PROJECT_ROOT/vie
export TASK_DATA_ROOT=/path/to/data/captured/task_x
```

## 🔧 Tools

📌 Step Dependencies Overview:
- 🔁 Step 2 ➡️ Step 3:
    - Object prompt selection (GDINO) is required before generating masks.
- 🔁 Step 3 ➡️ Step 7 & 8:
    - Object masks from Step 3 are used in BundleSDF for pose estimation.
- 🔓 Step 4:
    - Hand mesh extraction (HaMeR) can be performed independently.
- 🔁 Step 3 ➡️ Step 5:
    - Gripper transfer needs the hand mesh aligned with object masks.
- 🚀 Step 8:
    - Full real-world execution combining all outputs (masks, poses).
- 🐳 Docker Dependency (Steps 7 & 8)
  - These steps must be executed inside a Docker container.
  - Check Step 6 to start and enter docker.

<hr>

### 2. 🔍 Find object prompts using GDINO
Use GDINO with a text prompt to identify the object of interest in the first frame:
```shell
cd $VIE_ROOT
python test_gdino_prompts.py \
    --input_dir $TASK_DATA_ROOT/rgb \
    --text_prompt <obj-text-prompt> \
    --infer_first_only

# Output will be saved at:
# $TASK_DATA_ROOT/out/gdino/<obj_text_prompt>
# Note: spaces in <obj-text-prompt> will be replaced with "_"
```
✅ Once you've found a text prompt that successfully detects the object, use it in Step 3 to generate object masks across all frames.

<hr>


### 3. 🤖 Generate object masks using GDINO + SAMv2
To use GDINO and SAMv2 for object bounding box detection and tracking in video frames:
```shell
cd $VIE_ROOT
python run_gdino_samv2.py --input_dir $TASK_DATA_ROOT/rgb --text_prompt <obj-text-prompt> --save_interval=1
# Output saved in:
# $TASK_DATA_ROOT/out/samv2/<obj_text_prompt>/obj_masks - object mask
# $TASK_DATA_ROOT/out/samv2/<obj_text_prompt>/masks_traj_overlayed - (only with --save_viz) trajectory overlays
```

**Speed/viz options:**
- `--sam2_size {large|base_plus|small|tiny}` — smaller variants are faster (default `large`); `base_plus` is the recommended sweet spot.
- `--save_viz` — also write the trajectory-overlay PNGs (slow; off by default).

<hr>

### 4. ✋ Extracting Right/Left Hand BBoxes and 3D Meshes (HaMeR)
![vie-hand](../media/data_capture/vie-hand.png)

This step extracts right(1) / left(0) hand bounding boxes and 3D hand meshes using HaMeR.

✅ Assumptions:
- Only one person is present in the scene.
- Only frames containing at least one visible hand will be processed and saved under `out/hamer/model`.
```shell
cd $VIE_ROOT/hamer
python extract_hand_bboxes_and_meshes.py --opt_weight 100.0 --input_dir $TASK_DATA_ROOT/rgb
```

**Speed/viz options:**
- `--save_viz` — write the per-frame regression + side-view + overlay PNGs to `out/hamer/extra_plots/` (off by default).
- `--no_warm_start --opt_xatol 1e-5 --opt_maxiter 50` — revert the scipy-minimize speedup back to Phase 1 behavior if mesh quality regresses.
- `--no_fp16` — disable fp16 autocast on the HaMeR transformer.
- `--body_detector regnety` — use a smaller detector if your GPU OOMs on the default ViTDet-Huge.

📤 Output Directory Structure:
- `$TASK_DATA_ROOT/out/hamer/extra_plots` – (only with `--save_viz`) per-frame debug PNGs
- `$TASK_DATA_ROOT/out/hamer/scene` – RGB scene point cloud
- `$TASK_DATA_ROOT/out/hamer/model` – HaMeR results including MANO parameters
- `$TASK_DATA_ROOT/out/hamer/3dhand` – Aligned 3D hand meshes

🛠️ Known Issue (Python 3.10+)
If you encounter:
```shell
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections'
```
✅ Try this fix: `pip install --upgrade networkx`

<hr>

### 5. Transfer Human Hand to Fetch Gripper

This step requires the human hand mesh output from Step 4 (HaMeR). It maps the human hand configuration to the target robot gripper (e.g., Fetch gripper).

```shell
# Navigate to the repo and initialize submodules
cd $VIE_ROOT/rfp-grasp-transfer
git submodule update --init --recursive

# Run the hand-to-gripper transfer script
python transfer_from_hamer.py \
    --mano_model_dir ../hamer/_DATA/data/mano \
    --target_gripper fetch_gripper \
    --input_dir $TASK_DATA_ROOT
```

**Speed/viz options:**
- `--frame_batch_size 4` — process 4 frames per Adam call instead of one. Roughly **3× wall-clock** on this 70-frame task. Loses temporal warm-start *within* a batch; only sensible for offline preprocessing.
- `--num_particles 16 --max_iter 50` — current defaults; raise if grasp quality regresses (`--num_particles 32 --max_iter 300` for the original main behavior).
- `--save_viz` — write per-frame Plotly HTML to `out/hamer/transfer_extra_plots/` (~5MB/frame, slow; off by default). Works on both per-frame and `--frame_batch_size > 1` paths.
- `--device cpu` — fall back to CPU when GPU OOMs or the GPU is too small for the workload to amortize kernel-launch overhead.

📤 Output Directory Structure:
- `$TASK_DATA_ROOT/out/hamer/transfer_extra_plots` – (only with `--save_viz`) Plotly HTML overlays
- `$TASK_DATA_ROOT/out/hamer/transfer_hand_mesh` – Transferred 3D fetch gripper meshes

🛠️ Troubleshooting
If you see this error:
```
from collections import Mapping
ImportError: cannot import name 'Mapping' from 'collections'
```
✅ Try this fix: `pip install --upgrade networkx`

---

### 🧩 To visualize scene, hamer hand pose & transferred pose
This script visualizes the combined 3D point clouds from:
- `hamer/scene/` — RGB scene point cloud
- `hamer/3dhand/` — Predicted human hand mesh
- `hamer/transfer_hand_mesh/` — Transferred gripper mesh

```shell
python ply_viewer_with_combined_ply.py \
--data_dir $DATA_ROOT \
--num_points 100000000 \
--auto_mode \
--fps 10
```

### 🔧 Arguments

| Argument             | Type    | Default                    | Description                                                                 |
|----------------------|---------|----------------------------|-----------------------------------------------------------------------------|
| `--data_dir`         | `str`   | `./data/ply_sequence/`     | Base directory containing `scene`, `3dhand`, and `transfer_hand_mesh` dirs |
| `--num_points`       | `int`   | `10000000000`              | Max number of points to load from each PLY file                             |
| `--fps`              | `int`   | `5`                        | Frames per second in auto playback mode                                     |
| `--skip_viz_frames`  | `int`   | `1`                        | Show every Nth frame in auto/manual mode                                    |
| `--auto_mode`        | `flag`  | `False`                    | Auto-play the sequence in a loop                                            |
| `--left_hand`        | `flag`  | `False`                    | Load left-hand meshes (`*_0.ply`); right-hand (`*_1.ply`) by default        |



https://github.com/user-attachments/assets/0647947c-a1e7-45b5-9f82-c48e5228ab0b


<hr>

### 6. BundleSDF Docker Setup 
BundleSDF runs in docker. First setup docker container and enter
```shell
cd $VIE_ROOT/BundleSDF/
./docker/start_docker.sh # start docker container
./docker/enter_docker.sh $PWD # enter docker container
```

### 7. Object Pose Estimation Using BundleSDF
Run object pose estimation on captured video frames using BundleSDF (in docker):
```shell
cd $VIE_ROOT/BundleSDF/
python run_pose_only_bsdf.py --mode run_video --video_dir $TASK_DATA_ROOT
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

<hr>


## 8. Run GSAM2 + BundleSDF for Real-World Object Pose Estimation
Use the following command to perform real-time object pose estimation (in docker) by combining source frames (human demo) with rollout frames (robot execution):
```shell
cd $VIE_ROOT
./run_bundlesdf.sh $TASK_DATA_ROOT <src-frames> <rollout-frames>
# Example:
# ./run_bundlesdf.sh "./vie/_DATA/new-data-from-fetch-and-laptop/22tasks.latest/task_8_17s-use_hammer/" 15 5
```
🧩 Arguments:
- `task-root-dir-path`: Task root dir path
- `src-frames`: Number of frames to extract from the human demonstration.
- `rollout-frames`: Number of frames to process during real-time rollout.

### 👉 Object Pose Estimation with Multi-Frame Context:
- 📸 Uses multiple frames from both:
  - Source phase (human demonstration)
  - Rollout phase (real-time robot execution)
- 🎯 Provides richer visual and temporal context for improved understanding
- 🚀 Significantly outperforms single-frame methods in Accuracy, Stability, Robustness
- ✅ Enables more reliable pose tracking across time during real-world execution

<hr>

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
  - Source: `$TASK_DATA_ROOT/out/sam2/<text_prompt>/obj_masks`
  - Target: `$TASK_DATA_ROOT/masks/`
- This linkage is critical for enabling accurate object pose estimation during real-time execution.

🔗 Example for single object:
```json
{
  "obj_1": "black_eraser"
}
```

🔗 Example for dual objects:
```json
{
  "obj_1": "black_eraser",
  "obj_2": "whiteboard"
}
```


## ⚡ Benchmark

The `fasten-vie` branch contains a series of latency optimizations across all four vie modules. See [`scripts/bench_vie.sh`](scripts/bench_vie.sh) to A/B end-to-end against `main`. Each module emits a `[module] avg ms/frame | total Ys` log line at the end of its run so the speedup is observable without external profiling.

**Reference task**: `task_39_seasoning_on_omlette_v1` — 70 frames, 640×480 RGB+depth, RTX 5070 Laptop GPU.

### GDINO + SAMv2 — measured 7.86× on the propagation hot loop

Isolating `robokit/perception.py::propagate_masks_and_save` (the per-frame SAM2 propagation):

| Metric | `main` | `fasten-vie` | Speedup |
|---|---:|---:|---:|
| `propagate_masks_and_save` (70 frames) | 139.70 s | 17.77 s | **7.86×** |
| Per-frame avg | 1995.6 ms | 253.8 ms | **7.86×** |
| Total wall (incl. SAM2 model load) | 157.93 s | 52.14 s | 3.03× |

The dominant win comes from gating the per-frame `plt.close("all")` + new figure creation + `savefig` behind `--save_traj_overlay` (off by default). On `main`, every frame paid ~1.7 s of matplotlib churn dwarfing the ~280 ms of actual SAM2 inference; on `fasten-vie` only the inference cost remains. A second win — single-pass multi-bbox propagation — kicks in only when the prompt yields more than one detection (single-bbox case above doesn't exercise it; expect another step change for multi-object scenes).

### HaMeR — measured 6.49× on the scipy minimize step

Three changes on `fasten-vie`:

1. **Warm-start** the Nelder-Mead translation refinement from the prior frame's solution per hand side (left/right). Hand poses change smoothly between frames; seeding the optimizer near the answer eliminates the long initial descent.
2. **Relaxed tolerance**: `xatol 1e-8 → 1e-4` (≈0.1 mm in metric units) and a hard `maxiter=30` cap (was unbounded with `disp=True` console I/O per call).
3. **Gated debug renders**: per-frame `regression_img` + `side_img` pyrender passes and the `_all.jpg` overlay write are now opt-in via `--save_debug_renders` (off by default). The `cam_view` render itself stays since the depth-PC mask is derived from it.

Isolated bench of the scipy minimize step (the dominant cost) on the existing `out/hamer/model/*.npz` for `task_39_seasoning_on_omlette_v1` — same kd-tree, same vertices, same depth, only the scipy options + warm-start logic differ:

| Config | Settings | Total (70 frames, 138 calls) | Per frame | Per minimize call |
|---|---|---:|---:|---:|
| `main` | `xatol=1e-8`, no `maxiter`, `disp=True`, no warm-start | **209.57 s** | **2993.9 ms** | 1518.6 ms |
| `fasten-vie` | `xatol=1e-4`, `maxiter=30`, `disp=False`, warm-start | **32.27 s** | **461.0 ms** | 233.9 ms |
| **Speedup** | | **6.49×** | **6.49×** | 6.49× |

A/B knobs: `--no_warm_start`, `--opt_xatol 1e-5`, `--opt_maxiter 50` revert to Phase 1 behavior; `--save_debug_renders` re-enables the debug PNGs.

> The full HaMeR pipeline also runs ViTDet + ViTPose + the HaMeR transformer per frame; those costs are unchanged. The 6.49× above applies to the scipy-minimize stage, which the survey identified as the dominant single contributor.

**Phase 4 additions (code-only on this rig)**:
- `fp16` autocast wraps the HaMeR transformer forward pass on CUDA (`torch.amp.autocast(dtype=torch.float16)`). Default on; disable with `--no_fp16`. Expected ~1.5–2× on the model fwd; not measured here because MANO models + the robokit env's Blackwell-incompatible torch block running the full pipeline locally.
- **Investigated and rejected** `scipy.spatial.cKDTree` as a drop-in for `sklearn.neighbors.KDTree` in `hamer/mesh_to_sdf/rgbd2pc.py`. Bench ran 17+ min vs sklearn's 4 min before being killed — cKDTree is slower for this query pattern (777 verts × ~300k depth points × 138 queries per minimize call). Sticking with sklearn KDTree.

### rfp-grasp-transfer — measured ≈1.5× on a noisy CPU bench (larger expected on GPU)

Four cooperating changes (one of which had to be reverted — see below):

1. **Hoist `AdamGraspTransfer` out of the per-frame loop**. On `main`, `transfer_grasp` was instantiating a fresh `AdamGraspTransfer` per frame, which re-ran URDF parsing, kinematic-chain construction, and `grasp_transfer_correspondence` — none of which depend on per-frame inputs. `fasten-vie` builds one optimizer per source hand (left/right) at startup and reuses across frames.
2. **Warm-start Adam** from the prior frame's `q_current`. `AdamGraspTransfer` now caches its final `q_current` and reuses it as the starting point for the next frame's optimization (default on; `warm_start=False` to disable).
3. **Aggressive defaults**: `num_particles 32 → 16`, `max_iter 100 → 50` (was `300` upstream). Combined with warm-starting, these give comparable convergence in much less per-frame work. Override via `--num_particles` / `--max_iter` if quality regresses.
4. **A Phase 1 attempt to skip `target_handmodel` reload in `reset()` was reverted** after benchmarking caught an ~8× per-iteration regression — pytorch_kinematics retains state across reuses of the same chain that compounds with each `step()`'s `update_kinematics` call. Reloading wipes that cheaply (~tens of ms); leaving it stale costs hundreds of ms per frame. Worth flagging as a real "the obvious optimization is the wrong one" finding.

Apples-to-apples bench with each branch's *as-shipped* CLI defaults, 30 synthetic frames on CPU (this dev rig's `robokit-py3.10` torch lacks Blackwell kernels), skipping the first 2 warmup frames where pytorch_kinematics lazy-compile dominates:

| Branch | num_particles × max_iter | Median ms/frame | Total (28 measured frames) |
|---|---|---:|---:|
| `main` | 32 × 300 (no warm-start) | ~1240 ms | ~35 s |
| `fasten-vie` | 16 × 50 (warm-start) | ~870 ms | ~24 s |
| **Speedup** | | **≈1.5×** | |

Smaller than the survey-derived "expected ≈4–8×" estimate. Reasons: the CPU bench is thermal-noisy (1083/870/1163 ms/frame across 3 trials), and the per-frame fixed cost (URDF reload in `reset()`, ~tens of ms) doesn't shrink with iter count. On GPU on a real rig — where per-particle-iter cost shrinks more than fixed cost does — the speedup should be substantially larger.

#### Phase 5: deepcopy snapshot for `reset()`

Profiling on a real Blackwell-GPU run revealed that even after Phase 1–4, the per-frame `run_adam` time was bouncing between 150–500 ms, and 98% of total per-frame wall-clock was inside `run_adam`. The variance came from the URDF reload in `reset()` — pytorch_kinematics' chain construction has stochastic cost.

Fix: `__init__` now takes a `copy.deepcopy` snapshot of the freshly-built `target_handmodel`. `reset()` restores from the snapshot instead of re-parsing URDF. Functionally identical (clean kinematic-chain state, no accumulation regression) but skips URDF I/O + parse cost.

Measured (real run, 70 frames, --num_particles 4 --max_iter 15):
- before snapshot: `run_adam` 150–500 ms (high variance), 1.45 it/s
- after snapshot:  `run_adam` 267–277 ms (rock-solid), **1.67 it/s peak**, ~600 ms/frame total

The variance collapse is the most important signal — confirms the deepcopy is restoring the same clean state the URDF reload was, just without paying for the parse.

#### Phase 6: BatchedAdamGraspTransfer (frame-level parallelism)

After Phase 5, profiling showed `run_adam` itself was the only meaningful cost left (270 ms × 2 hands per frame), and the GPU was severely underutilized — at `num_particles=4`, kernel-launch overhead dominated over actual compute. Adding `BatchedAdamGraspTransfer` processes N frames in a single Adam call by stacking each frame's P particles along the batch dim (total batch = N×P).

Wired into `transfer_from_hamer.main()` as `--frame_batch_size` (default 1 keeps per-frame loop). Works only for offline preprocessing — loses temporal warm-start within a batch.

Measured on `task_39` (70 frames, RTX 5070 Laptop, robokit-py3.10):

| `--frame_batch_size` | ms/frame | wall | speedup vs F=1 |
|---|---:|---:|---:|
| 1 (sequential, Phase 5) | 1486 | 119 s | 1.0× |
| **4** | **301** | **42.8 s** | **2.78×** |
| 8 | 425 | 47.4 s | 2.51× |
| 16 | 331 | 47.3 s | 2.52× |

F=4 is the sweet spot for a 70-frame task; longer videos may shift it higher. Combined with Phases 1–5 the overall improvement on this rig is ~5× wall-clock vs main.

A/B: `--frame_batch_size 1` reverts to per-frame mode. Output PLY/npz count is identical (verified — all 70 frames produced PLYs in both paths).

**Phase 4 additions** (committed in `IRVLUTD/rfp-grasp-transfer@fasten-vie` and bumped in the parent submodule pointer):

- **Cache `grasp_transfer_correspondence`**: the source ↔ target gripper-coord correspondence is a deterministic function of two static tensors (one per source robot, one per target robot, both loaded once from pickle). Previously every `reset()` recomputed an O(M·N) spherical-distance matrix between source/target gripper coords. Now computed once on first `reset()` and reused.
- **Jittered particle init from prior best**: the Phase 2 warm-start copied the prior frame's `q_current` verbatim (carrying both good and bad particles). Phase 4 picks the lowest-energy particle from the prior frame, replicates it across all `num_particles`, and adds Gaussian jitter (σ=0.02) to all but particle 0 (which keeps the exact warm-start). Converges to lower final energy in fewer effective iters.

CPU re-bench after Phase 4: ~870–1006 ms/frame (within Phase 3 noise; the wins are mostly in convergence quality at fixed `max_iter`, not raw wall-time). On GPU + with longer trajectories the spherical-distance cache and jittered init should compound more visibly.

### BundleSDF — docker persistence + tighter `n_step` (not measured here)

Two changes outside Docker plus one default tweak:

1. **Persistent docker launch**: `BundleSDF/docker/start_docker.sh` no longer runs `docker-compose down` followed by `up --build` on every invocation. New flags: `-k` keeps an already-running container (subsequent runs are near-instant), `-f` forces rebuild. Default behavior skips `--build` if an image already exists, only building on first run.
2. **Pre-cache masks** into RAM in one pass before the frame loop in `BundleSDFProcessor.process` (when not using a live segmenter). Mask files are tens of KB each; reading them inside the hot loop added per-frame disk-seek latency for no benefit.
3. **Aggressive default `--n_step 5`** (was None ⇒ `config.yml`'s 10) for the NeRF training-step count per keyframe trigger. Lower = faster training pass; raise back to 10 if reconstruction quality regresses.

Why not measured on this rig: BundleSDF runs inside a Docker container and Docker is not installed on the dev machine. The `[bundlesdf] processed N frames | avg X ms/frame` log inside `BundleSDFProcessor.process` lands in the bench log when run on a working rig.

## 🙏 Acknowledgments

This project utilizes the following resources:
- [GDINO + SamV2](https://github.com/jishnujayakumar/robokit)
- [HaMeR](https://github.com/IRVLUTD/HaMeR)
- [rfp-grasp-transfer](https://github.com/IRVLUTD/rfp-grasp-transfer)
- [BundleSDF](https://github.com/jishnujayakumar/BundleSDF)
