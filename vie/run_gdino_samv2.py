# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
# ----------------------------------------------------------------------------------------------------

"""
This script performs object tracking on video frames using Grounding DINO for initial detection
and SAM2 for tracking each detected bounding box across frames.

The script follows these steps:

1. **Initialize Flags**:
   - Define command-line flags for the input directory containing video frames, a text prompt for object detection,
     and an interval for saving frames during tracking.

2. **Main Function**:
   - Parse the command-line flags to obtain the input directory, text prompt, and save interval.
   - Ensure both input directory and text prompt are provided; otherwise, raise an error.

3. **Setup Object Detectors**:
   - Initialize the Grounding DINO object detector to detect bounding boxes in the first frame based on the given text prompt.
   - Initialize SAM2 to track detected bounding boxes across all frames.

4. **Process First Frame**:
   - Load the first frame from the input directory.
   - Use Grounding DINO to detect initial bounding boxes with the specified text prompt.

5. **Track All Detected Bounding Boxes**:
   - For each bounding box detected in the first frame:
     a. Convert the bounding box to a [x_min, y_min, x_max, y_max] format.
     b. Pass each bounding box to SAM2 to propagate the bounding box across all frames, tracking the object's movement.
     c. Set the tracking save interval to the specified value for periodic saving of the tracking results.

6. **Output**:
   - Each bounding box's tracking results are saved in the specified save interval.
   - Log messages indicate the start and completion of tracking for each bounding box.

**Usage**:
   - Run this script from the command line, passing the required `input_dir` and `text_prompt` flags,
     and an optional `save_interval` (default is 1):

     `python run_gdino_samv2.py --input_dir=<path_to_frames> --text_prompt="object description" --save_interval=2`
"""


import os
# Silence the third-party deprecation/registry/dep-version chatter that fires
# on every import (timm.models.layers, mobile_sam tiny_vit registry, transformers
# device-arg deprecations, torch checkpoint use_reentrant, requests urllib3
# version mismatch, etc.). Real exceptions still propagate.
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Lightweight imports first so we can show a banner immediately. The heavy ML
# stack (torch / GroundingDINO / SAM2 / robokit.perception) takes ~5 s to
# import on a fresh process; without a banner the user stares at a blank
# terminal during that time.
import contextlib
import io
import sys
import time as _time
from robokit import log as vlog

vlog.section("GDINO + SAMv2 — object mask propagation")
with vlog.working("Importing ML stack (torch, GroundingDINO, SAM2, robokit)"):
    import numpy as np
    from absl import app, flags
    from PIL import Image as PILImg
    from robokit.perception import GroundingDINOObjectPredictor, SAM2VideoPredictor

    # absl + downstream lib loggers are too verbose for end-user runs.
    import logging as _stdlogging
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.WARNING)  # absl uses its own setup
    _stdlogging.getLogger("absl").setLevel(_stdlogging.WARNING)
    for _noisy in ("sam2", "build_sam", "groundingdino", "transformers"):
        _stdlogging.getLogger(_noisy).setLevel(_stdlogging.WARNING)
    _stdlogging.getLogger().setLevel(_stdlogging.WARNING)

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory path to input video frames")
flags.DEFINE_string("text_prompt", None, "Text prompt for initial object detection")
flags.DEFINE_integer("save_interval", 1, "Interval for saving tracked frames")
flags.DEFINE_boolean(
    "save_traj_overlay",
    False,
    "Also save matplotlib trajectory-overlay PNGs (slow; off by default).",
)
flags.DEFINE_boolean(
    "save_viz",
    False,
    "Unified across vie scripts: save all visualization/debug artifacts for "
    "this stage. Equivalent to --save_traj_overlay here.",
)
flags.DEFINE_string(
    "sam2_size",
    "large",
    "SAM2.1 model size variant: large (default, best quality) | base_plus | small | tiny. "
    "Smaller = faster forward pass; checkpoint auto-downloaded on first use.",
)


def _silent():
    """Swallow noisy stdout/stderr from third-party model loads (BERT
    text_encoder_type print, GroundingDINO incompatible-keys log, etc.)."""
    return contextlib.redirect_stdout(io.StringIO())


def main(argv):
    video_dir, text_prompt = sanity_check(argv)
    save_interval = FLAGS.save_interval
    if not video_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    log = vlog.setup("gsam")
    t_start = _time.time()

    # ---------------- Configuration ----------------
    vlog.section("Configuration")
    vlog.note(f"input dir       : {video_dir}")
    vlog.note(f"text prompt     : {text_prompt!r}")
    vlog.note(f"sam2 model size : {FLAGS.sam2_size}")
    vlog.note(f"save interval   : {save_interval}")
    save_traj = bool(FLAGS.save_traj_overlay or FLAGS.save_viz)
    vlog.note(f"save overlay    : {save_traj} "
              f"(--save_traj_overlay={FLAGS.save_traj_overlay}, --save_viz={FLAGS.save_viz})")

    try:
        # ---------------- Model loading ----------------
        vlog.section("Loading models")
        with vlog.working("Loading GroundingDINO (BERT + Swin-T + ckpts)"):
            with _silent():
                gdino = GroundingDINOObjectPredictor()
        with vlog.working(f"Loading SAM2.1 ({FLAGS.sam2_size}) — first run downloads checkpoint"):
            with _silent():
                sam2 = SAM2VideoPredictor(text_prompt, model_size=FLAGS.sam2_size)

        # ---------------- Detection ----------------
        vlog.section("Detection (first frame)")
        first_frame_path = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
        first_frame = PILImg.open(first_frame_path).convert("RGB")
        vlog.note(f"frame: {os.path.basename(first_frame_path)} ({first_frame.size[0]}×{first_frame.size[1]})")

        t = _time.time()
        with vlog.working(f"Running GroundingDINO with prompt {text_prompt!r}"):
            with _silent():
                initial_bboxes, _, _ = gdino.predict(first_frame, text_prompt)
        det_dt = _time.time() - t

        if len(initial_bboxes) == 0:
            vlog.error(f"No bounding boxes detected for prompt {text_prompt!r}.")
            return
        vlog.step(f"  ↳ {len(initial_bboxes)} bbox(es) detected")

        bbox_arrays = [
            np.array(gdino.bbox_to_scaled_xyxy(bbox, *first_frame.size))
            for bbox in initial_bboxes
        ]

        # ---------------- Propagation ----------------
        vlog.section("Tracking (SAM2 propagation)")
        vlog.note(
            f"propagating {len(bbox_arrays)} obj across all frames; "
            "init_state warms up first (DALI fast-path on if installed)…"
        )
        # SAM2 emits its own tqdm bar inside propagate_in_video — that's the
        # primary live progress signal during this stage; we just time the
        # whole call so the summary panel shows total wall + fps.
        t = _time.time()
        frame_names, video_segments = sam2.propagate_masks_and_save_multi(
            video_dir,
            bbox_arrays,
            save_interval,
            save_traj_overlay=save_traj,
        )
        prop_dt = _time.time() - t
        n_frames = len(frame_names)
        vlog.step(
            f"propagated {len(bbox_arrays)} obj × {n_frames} frames in "
            f"{vlog.fmt_duration(prop_dt)} ({vlog.fmt_rate(n_frames, prop_dt)})"
        )

        # ---------------- Summary ----------------
        out_root = os.path.join(
            os.path.dirname(video_dir), "out", "samv2",
            text_prompt.lower().replace(" ", "_"),
        )
        total_dt = _time.time() - t_start
        vlog.summary({
            "objects":      str(len(bbox_arrays)),
            "frames":       str(n_frames),
            "detection":    vlog.fmt_duration(det_dt),
            "propagation":  vlog.fmt_duration(prop_dt),
            "total wall":   vlog.fmt_duration(total_dt),
            "fps":          vlog.fmt_rate(n_frames, prop_dt),
            "output":       out_root,
        }, title="GSAM run summary")
        vlog.success("Done.")
    except Exception as e:
        vlog.error(f"Pipeline failed: {e}")
        log.exception("traceback")
        sys.exit(1)


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt

    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(
            f"Error: The directory '{input_dir}' does not exist or is not a valid directory."
        )

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise Exception(f"No image files found in the directory '{input_dir}'.")

    # Ensure text_prompt is provided
    if not text_prompt:
        raise Exception("Error: 'text_prompt' is required but not provided.")

    return input_dir, text_prompt


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required("input_dir")
    flags.mark_flag_as_required("text_prompt")

    # Run the main function
    app.run(main)
