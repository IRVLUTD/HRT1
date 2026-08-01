#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
This script is modified from https://github.com/geopavlakos/hamer/blob/df533a2d04b9e2ece7cf9d6cbc6982e140210517/demo.py
The script in its default state is capable of providing hand mesh for multiple persons in the image.
We have not modified this ability. In our setup, we make sure that one scene contains only one person
"""

import os
import sys
import time
import threading
import warnings


# --- Immediate user feedback before heavy imports kick in -------------------
# torch + detectron2 + mmcv + HaMeR + ViTPose imports take ~10 s on cold cache.
# Spawn a tiny spinner thread that writes directly to fd 2 so the user sees
# something happening within ~80 ms of pressing Enter. The fd-write bypasses
# any later `sys.stdout` / `sys.stderr` redirects we apply for noise control.
_IMPORTS_DONE = threading.Event()
_IMPORT_T0 = time.time()


def _import_spinner():
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    # Heavy ML stack is now deferred to HandInfoExtractor.__init__, so module
    # load is fast (~1-2 s). Keep the spinner just in case torch/cv2/scipy take
    # a moment on a cold machine.
    msg = "Starting up (torch, cv2, scipy)…"
    i = 0
    while not _IMPORTS_DONE.wait(0.08):
        line = f"\r\033[36m{chars[i % len(chars)]}\033[0m {msg} \033[2m({time.time() - _IMPORT_T0:0.1f}s)\033[0m"
        try:
            os.write(2, line.encode("utf-8"))
        except Exception:
            break
        i += 1


_spin = threading.Thread(target=_import_spinner, daemon=True)
_spin.start()

# --- Silence noisy third-party imports so the rich UI is the first thing the
# user sees. `simplefilter("ignore")` is stronger than filterwarnings — it
# replaces the filter list, so subsequent `simplefilter("default")` calls from
# transformers/mmcv/etc. can't reset us. Real errors still surface (tracebacks
# go to stderr; we only redirect stdout in tightly-scoped blocks below).
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import io
import re
import cv2
import torch
import contextlib
import logging
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
# open3d only used in save_point_cloud_as_ply (per-frame call); lazy-import to
# skip ~1-2s of startup when running --help or short test cases.
# import matplotlib.pyplot as plt

# Re-arm the simplest filter after `import torch` — torch's logging setup can
# install its own filter on first import.
warnings.simplefilter("ignore")


@contextlib.contextmanager
def _silence_io(stderr: bool = True):
    """Suppress raw stdout (and optionally stderr) writes from noisy C-extension
    third-party libs that bypass the Python warnings system. Used around model
    loading + heavy imports — short-lived so any real fatal error still surfaces
    via the Python exception path (which prints traceback to stderr via the
    interpreter, *after* our context has exited)."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(buf_out))
    if stderr:
        stack.enter_context(contextlib.redirect_stderr(buf_err))
    try:
        with stack:
            yield
    except Exception:
        # On error, write the captured buffers out so the user can see what
        # the third-party module was trying to say.
        sys.stdout.write(buf_out.getvalue())
        sys.stderr.write(buf_err.getvalue())
        raise


# Heavy ML imports (mmcv/mmpose/HaMeR/pyrender) are *deferred* — pulling them
# in at module load wastes ~10 s of startup before the user sees anything. We
# bind None placeholders at module scope and populate them via
# `_ensure_heavy_imports()` the first time `HandInfoExtractor()` is built.
# Subsequent calls are no-ops.
RGBD2PC = None
recursive_to = None
CACHE_DIR_HAMER = None
download_models = None
load_hamer = None
DEFAULT_CHECKPOINT = None
ViTDetDataset = None
DEFAULT_MEAN = None
DEFAULT_STD = None
Renderer = None
cam_crop_to_full = None
ViTPoseModel = None
_heavy_imports_done = False


def _ensure_heavy_imports():
    """Populate the module-level placeholders with the real symbols, split into
    four per-library spinners so the user sees which import block is slow:
      1. HaMeR / transformers / smplx
      2. detectron2 / mmcv (via hamer.datasets.vitdet_dataset)
      3. pyrender / trimesh (via hamer.utils.renderer)
      4. mmpose / ViTPose
    """
    global _heavy_imports_done
    global RGBD2PC, recursive_to, CACHE_DIR_HAMER
    global download_models, load_hamer, DEFAULT_CHECKPOINT
    global ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    global Renderer, cam_crop_to_full, ViTPoseModel
    if _heavy_imports_done:
        return

    with vlog.working("Importing HaMeR + transformers + smplx"):
        with _silence_io():
            from mesh_to_sdf.rgbd2pc import RGBD2PC as _RGBD2PC
            from hamer.utils import recursive_to as _recursive_to
            from hamer.configs import CACHE_DIR_HAMER as _CACHE_DIR_HAMER
            from hamer.models import download_models as _download_models
            from hamer.models import load_hamer as _load_hamer
            from hamer.models import DEFAULT_CHECKPOINT as _DEFAULT_CHECKPOINT
        RGBD2PC = _RGBD2PC
        recursive_to = _recursive_to
        CACHE_DIR_HAMER = _CACHE_DIR_HAMER
        download_models = _download_models
        load_hamer = _load_hamer
        DEFAULT_CHECKPOINT = _DEFAULT_CHECKPOINT

    with vlog.working("Importing detectron2 + mmcv (ViTDet)"):
        with _silence_io():
            from hamer.datasets.vitdet_dataset import ViTDetDataset as _ViTDetDataset
            from hamer.datasets.vitdet_dataset import DEFAULT_MEAN as _DEFAULT_MEAN
            from hamer.datasets.vitdet_dataset import DEFAULT_STD as _DEFAULT_STD
        ViTDetDataset = _ViTDetDataset
        DEFAULT_MEAN = _DEFAULT_MEAN
        DEFAULT_STD = _DEFAULT_STD

    with vlog.working("Importing pyrender + trimesh"):
        with _silence_io():
            from hamer.utils.renderer import Renderer as _Renderer
            from hamer.utils.renderer import cam_crop_to_full as _cam_crop_to_full
        Renderer = _Renderer
        cam_crop_to_full = _cam_crop_to_full

    with vlog.working("Importing mmpose + ViTPose"):
        with _silence_io():
            from vitpose_model import ViTPoseModel as _ViTPoseModel
        ViTPoseModel = _ViTPoseModel

    _heavy_imports_done = True


# Shared rich-based UI used by the other vie scripts (gsam2, rfp).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robokit import log as vlog  # noqa: E402

from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List

# Module-level imports done (the cheap ones). Stop the import spinner — the
# heavy ML stack import has been moved into HandInfoExtractor.__init__ where
# the rich "Loading models" spinner will cover it.
_IMPORTS_DONE.set()
_spin.join(timeout=0.5)
try:
    _elapsed = f"{time.time() - _IMPORT_T0:.1f}s"
    os.write(2, f"\r\033[K\033[32m✓\033[0m Module ready \033[2m({_elapsed})\033[0m\n".encode("utf-8"))
except Exception:
    pass

# Quiet the default logging — vlog.setup() configures the rich handler instead.
logger = logging.getLogger(__name__)


def _model_cache_path(checkpoint_path: str, body_detector: str) -> Optional[str]:
    """Return cache path for the assembled detector+cpm models, or None if we
    can't compute a stable fingerprint (e.g. no checkpoint file)."""
    import hashlib
    try:
        ckpt_stat = os.stat(checkpoint_path)
        fingerprint = hashlib.sha256(
            f"{checkpoint_path}|{ckpt_stat.st_size}|{ckpt_stat.st_mtime_ns}"
            f"|{body_detector}|v1".encode()
        ).hexdigest()[:16]
    except OSError:
        return None
    cache_dir = os.path.expanduser("~/.cache/hamer")
    return os.path.join(cache_dir, f"detector_cpm_{fingerprint}.pt")


def _try_load_model_cache(path: str) -> Optional[dict]:
    """Try to load the detector+cpm cache. Silent fallback to fresh build on
    any error — pickle/torch.load is fragile across versions, the only safe
    response is to rebuild."""
    if not path or not os.path.exists(path):
        return None
    try:
        with _silence_io():
            cached = torch.load(path, weights_only=False, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(cached, dict) or 'detector' not in cached or 'cpm' not in cached:
            return None
        return cached
    except Exception:
        # Stale or incompatible cache; let the caller build fresh.
        return None


def _save_model_cache(path: str, detector, cpm) -> None:
    """Save detector+cpm to disk for next-run reuse. Silent on failure."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _silence_io():
            torch.save({'detector': detector, 'cpm': cpm}, path)
    except Exception:
        # No cache, but the run already succeeded — keep silent.
        pass

# Constants
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


@dataclass
class ExtractorOutput:
    hand_boxes: np.ndarray
    is_right: np.ndarray
    output_path: Optional[List[str]] = None
    pcd: Optional[List[object]] = None
    out: Optional[object] = None
    opt_weight: Optional[float] = 100.0


@dataclass
class OptConfig:
    # Aggressive defaults: typical hand poses change smoothly between frames so
    # warm-starting from the previous solution converges in a few dozen Nelder-Mead
    # steps to ~mm-scale accuracy. xatol=1e-4 corresponds to ~0.1 mm in metric units.
    xatol: float = 1e-4
    maxiter: int = 30
    disp: bool = False
    save_debug_renders: bool = False
    warm_start: bool = True
    # fp16 autocast on the HaMeR transformer forward pass. Uses torch.amp on
    # CUDA devices; no-op on CPU. ~1.5-2x speedup on the model fwd in practice
    # with no observable mesh-quality regression. Set False to fall back to fp32.
    fp16: bool = True
    # Body-detector stride. ViTDet-Huge runs at ~700 ms/frame and dominates wall-
    # clock when invoked every frame. For typical demos with one stationary or
    # slow-moving demonstrator, the body bbox is ~constant; rerunning every K
    # frames and reusing the cached bbox in between is safe. We auto-redetect
    # when the cached bbox produces no valid hand keypoints (the natural
    # safety net — bbox is wrong, ViTPose fails, we retry on a fresh detect).
    detector_stride: int = 5
    # Scene PLY (out/hamer/scene/*.ply) is only consumed by scripts/tools/seq_viewer.py;
    # rfp-grasp-transfer never reads it. Off by default — saves ~3 s/frame.
    save_scene_pcd: bool = False
    # 3dhand PLY (out/hamer/3dhand/*.ply) is the per-hand point cloud aligned to
    # the real camera. rfp-grasp-transfer reads it ONLY in its viz/--save_viz
    # path (transfer_from_hamer.py:382 inside `if debug_plots:`); production rfp
    # computes everything from model/*.npz instead. The other consumers are
    # scripts/tools/seq_viewer.py and create_combined_ply.py — also viz-only.
    # Off by default — saves ~45 ms/frame on this rig.
    save_3dhand_pcd: bool = False
    # Translation-refinement backend. "torch" runs an Adam loop on GPU with
    # cdist-based 3D nearest-neighbor; ~5-10x faster than scipy Nelder-Mead on
    # this rig and trivially batches across both hands within a frame.
    minimize_backend: str = "torch"  # one of {"torch", "scipy"}
    # `torch_steps` is the *upper bound* on Adam iterations. With early-stop on,
    # most warm-started frames converge in 15-25 steps; cold frames burn the
    # full 50. Keep this at 50 to preserve quality on hard frames.
    torch_steps: int = 50
    torch_lr: float = 5e-3
    # Early-stop knobs. After at least `torch_min_steps` iterations, break the
    # Adam loop when the relative loss change drops below `torch_tol`. Disabled
    # by setting torch_tol=0.
    torch_min_steps: int = 15
    torch_tol: float = 1e-3
    # Hand-mask backend. The mask is only used to *exclude* hand-surface depth
    # points from the depth_pc that the translation optimization fits against.
    # 'pyrender' is the upstream renderer (~370 ms/frame, 42% of wall on this
    # rig). 'fast' projects MANO verts via K1 and fills the convex hull on a
    # numpy buffer in cv2 — ~5 ms/frame. The mask is *looser* than the rendered
    # alpha (no z-cull, no triangle-level edges) but the optimization only
    # cares about excluding the hand region, which the convex hull does.
    mask_backend: str = "fast"  # one of {"fast", "pyrender"}
    # Cross-frame GPU batching. When > 1, buffer K frames worth of per-frame
    # state (detector → ViTPose → HaMeR fwd → mask → depth_pc), then run ONE
    # batched torch Adam over all (K × hands_per_frame) at once. Saves Python
    # + kernel-launch overhead on the minimize stage. 1 = unchanged per-frame
    # behavior. Quality risk: within-chunk warm-start gets coarser (all frames
    # in the chunk start from the same x0 from the prior chunk).
    frame_batch_size: int = 1
    # Fixed depth-pc size for the batched-minimize path. Each frame's
    # depth_pc is subsampled (or replica-padded) to this many points so cdist
    # can run as one (H, 778, N_sub) call. 5000 was the sweet spot in the
    # micro-bench — bigger gives no quality gain, smaller starts losing NN
    # fidelity. Only used when frame_batch_size > 1.
    batched_depth_subsample: int = 5000


def load_depth_img(img_path):
    """
    Loads a depth image corresponding to the given RGB image path.

    This function replaces the 'rgb' directory with 'depth' and the file 
    extension from '.jpg' to '.png' to locate the depth image. It reads the 
    depth image, normalizes it by dividing by 1000 (to convert the depth 
    values from millimeters to meters), and returns the depth data as a 
    NumPy array.

    Source: https://github.com/IRVLUTD/hamer-depth/commit/070886168e469ab1645612a2c3b8c6473aab1aef#diff-6bacd8700314864adb2bf1d56bb841dab8e0ac87d88c8303caa83b545d0b4b9dR116

    Args:
        img_path (str): Path to the RGB image file.

    Returns:
        np.ndarray: Normalized depth image as a NumPy array.

    Raises:
        FileNotFoundError: If the depth image file does not exist.
        ValueError: If the depth image cannot be loaded or is invalid.
    """
    try:
        # Replace 'rgb' with 'depth' and change the extension to '.png'
        depth_path = str(img_path).replace('rgb', 'depth').replace('jpg', 'png')
        # logger.info(f"Attempting to load depth image from: {depth_path}")

        # Read the depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to load depth image from {depth_path}")

        # Convert depth to float32 and normalize
        depth = depth.astype(np.float32) / 1000.0
        # logger.info("Depth image loaded and normalized successfully.")

        return depth

    except FileNotFoundError as e:
        logger.error(f"Depth image file not found: {e}")
        raise

    except ValueError as e:
        logger.error(f"Error loading depth image: {e}")
        raise

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the depth image: {e}")
        raise


def obj_function(x, vertices, translation, K1, K2, kd_tree, weight_3d=10):
    """
    Computes the objective function value for optimization.

    This function calculates the combined error, which includes:
    - 2D projection error between two sets of 3D vertices projected onto 
      camera planes using intrinsic matrices.
    - Weighted 3D distance error between the translated vertices and 
      their nearest neighbors in a KD-tree.

    Args:
        x (np.ndarray): A 3D translation vector for the second projection (shape: (3,)).
        vertices (np.ndarray): Nx3 array of 3D points (vertices).
        translation (np.ndarray): A 3D translation vector for the first projection (shape: (3,)).
        K1 (np.ndarray): 3x3 intrinsic matrix for the first camera.
        K2 (np.ndarray): 3x3 intrinsic matrix for the second camera.
        kd_tree (scipy.spatial.KDTree): KD-tree for nearest-neighbor search.
        weight_3d (float, optional): Weight factor for the 3D error. Defaults to 100.

    Returns:
        float: The combined error value (2D projection error + weighted 3D error).
    """
    try:
        # Projection 1
        V1 = vertices + translation
        x1 = K1 @ V1.T
        x1[0, :] /= x1[2, :]
        x1[1, :] /= x1[2, :]
        # logger.debug("Projection 1 completed.")

        # Projection 2
        V2 = vertices + x
        x2 = K2 @ V2.T
        x2[0, :] /= x2[2, :]
        x2[1, :] /= x2[2, :]
        # logger.debug("Projection 2 completed.")

        # Compute 3D distances
        distances, _ = kd_tree.query(V2)
        distances = distances.astype(np.float32).reshape(-1)
        error_3d = np.mean(distances)
        # logger.debug(f"3D distance error: {error_3d}")

        # Compute 2D projection error
        error_2d = np.square(x1[:2] - x2[:2]).mean()
        # logger.debug(f"2D projection error: {error_2d}")

        # Combine errors with weighting
        total_error = error_2d + weight_3d * error_3d
        # logger.info(f"Total error: {total_error} (2D: {error_2d}, 3D: {error_3d}, Weight: {weight_3d})")

        return total_error

    except Exception as e:
        logger.error(f"An error occurred while computing the objective function: {e}")
        raise


class HandInfoExtractor:
    def __init__(self, checkpoint: Optional[str] = None, body_detector: str = 'vitdet', rescale_factor: float = 2.0, opt_cfg: Optional[OptConfig] = None, parallel_load: bool = False, use_model_cache: bool = True):
        # Each model gets its own spinner so the user sees per-step progress
        # instead of one ~45s blob labeled "Loading HaMeR + ViTDet + ViTPose".
        # The `_silence_io` wraps suppress the noisy state-dict / MANO-warning
        # prints from third-party loaders without affecting the rich spinners
        # (Console.file was snapshot at first use; see robokit/log.py).
        # `_ensure_heavy_imports()` owns its own four sub-spinners (HaMeR,
        # detectron2, pyrender, mmpose).
        _ensure_heavy_imports()

        if checkpoint is None:
            checkpoint = DEFAULT_CHECKPOINT
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with _silence_io():
            download_models(CACHE_DIR_HAMER)

        # HaMeR's LightningModule holds a ctypes pointer in smplx's MANO wrapper
        # → torch.save fails on it. So HaMeR is always loaded fresh from ckpt.
        # The detector + cpm pickle cleanly and get cached on disk.
        # NOTE: we DEFER the actual cache load until after HaMeR is on the GPU.
        # Loading the 5 GB cache first fragments / saturates GPU memory and the
        # subsequent HaMeR allocation fails on warm-cache runs.
        cache_path = _model_cache_path(checkpoint, body_detector) if use_model_cache else None
        has_cache = bool(cache_path and os.path.exists(cache_path))

        # --- HaMeR first while GPU is empty (always fresh from disk) ---
        with vlog.working("Loading HaMeR    (2.7 GB ckpt)"):
            with _silence_io():
                self.model, self.model_cfg = load_hamer(checkpoint)
                self.model = self.model.to(self.device).eval()

        # --- Now ViTDet + ViTPose (cache-aware, GPU has room) ---
        cached = None
        if has_cache:
            with vlog.working("Loading ViTDet + ViTPose from cache (~5 GB pickle)"):
                cached = _try_load_model_cache(cache_path)
                if cached is not None:
                    self.detector = cached['detector']
                    self.cpm = cached['cpm']
        if cached is None and parallel_load:
            # Threaded path — kept as escape hatch; on this rig it's slower
            # than sequential because the loaders are GIL-bound.
            import concurrent.futures as _cf
            with vlog.working("Loading ViTDet + ViTPose (parallel)"):
                with _silence_io(), _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="hamer-init") as _pool:
                    f_detector = _pool.submit(self._initialize_detector, body_detector)
                    f_cpm = _pool.submit(ViTPoseModel, self.device)
                    self.detector = f_detector.result()
                    self.cpm = f_cpm.result()
        elif cached is None:
            with vlog.working("Loading ViTDet   (600 MB ckpt)"):
                with _silence_io():
                    self.detector = self._initialize_detector(body_detector)
            with vlog.working("Loading ViTPose  (1.4 GB ckpt)"):
                with _silence_io():
                    self.cpm = ViTPoseModel(self.device)

        # --- Cache save (only on fresh build) ---
        if cache_path is not None and cached is None and not has_cache:
            with vlog.working("Writing model cache to disk (~5 GB)"):
                _save_model_cache(cache_path, self.detector, self.cpm)
        # Renderer is lazy-loaded — only the viz path (debug renders) and the
        # 'pyrender' mask backend ever construct it. The default processing path
        # (fast convex-hull mask) skips the Renderer entirely, including its
        # offscreen GL context setup at import time.
        self._renderer = None
        self.rescale_factor = rescale_factor
        self.opt_cfg = opt_cfg or OptConfig()
        # Warm-start cache: previous frame's optimized translation, keyed by hand side
        # (0 = left, 1 = right). Hand poses change smoothly so this seeds the
        # Nelder-Mead near the answer and lets us drop maxiter aggressively.
        self._last_opt_translation: dict = {}
        # Body-detector bbox cache for `detector_stride > 1`: stores the most recent
        # detector output along with the frame index it was produced on. We reuse
        # det_out for `stride - 1` subsequent frames, then redetect.
        self._cached_det_out = None
        self._cached_det_frame_idx: int = -10**9
        # Detector run/skip counters surfaced in the final summary panel.
        self.detector_runs = 0
        self.detector_skips = 0
        # Per-task cam_K cache — was np.loadtxt'd every frame. Lazy: keyed by the
        # cam_K.txt path so the same extractor can process multiple task dirs.
        self._cam_K_cache: dict = {}
        # Background NPZ writer — disk I/O for save_output_as_npz overlaps with
        # the next frame's GPU work instead of blocking the main loop.
        from concurrent.futures import ThreadPoolExecutor
        self._writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hamer-writer")
        self._pending_writes: list = []
        # Pre-allocated Adam state (rebuilt only when batch shape changes) and
        # a reusable depth tensor buffer. Cuts ~10-15 ms/frame of allocator churn.
        self._adam_optim: Optional[torch.optim.Adam] = None
        self._adam_t: Optional[torch.Tensor] = None
        self._depth_buf: Optional[torch.Tensor] = None
        # Cross-frame batching buffer. When `frame_batch_size > 1`, `_save_meshes`
        # gathers per-frame state up to (and including) depth_pc, appends it here,
        # and returns early. The main loop calls `flush_chunk()` every K frames
        # to run one batched torch Adam over all hands × K frames.
        self._chunk_buffer: list = []

    def _initialize_detector(self, body_detector: str):
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

    @property
    def renderer(self):
        """Lazy pyrender Renderer. Only constructed when the viz / pyrender-mask
        path actually needs it. Skipping this in processing mode avoids the
        offscreen GL context setup."""
        if self._renderer is None:
            self._renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        return self._renderer

    def _run_detector(self, img_cv2, frame_idx: int):
        det_out = self.detector(img_cv2)
        self._cached_det_out = det_out
        self._cached_det_frame_idx = frame_idx
        self.detector_runs += 1
        return det_out, False

    def _get_detector_output(self, img_cv2, frame_idx: int, stride: int):
        if (
            stride > 1
            and self._cached_det_out is not None
            and (frame_idx - self._cached_det_frame_idx) < stride
        ):
            self.detector_skips += 1
            return self._cached_det_out, True
        return self._run_detector(img_cv2, frame_idx)

    def _minimize_translation_torch_batched(self, frames_data, weight_3d: float):
        """Cross-frame batched Adam minimize.

        `frames_data` is a list of per-frame dicts each carrying:
            verts_list  : list of (N, 3) numpy arrays — one per hand
            cam_t_list  : list of (3,)  numpy arrays — one per hand
            K1          : (3, 3) HaMeR cam K
            K2          : (3, 3) real cam K (intrinsic_matrix)
            depth_points: (M_i, 3) per-frame filtered depth point cloud
            x0_list     : list of (3,) — warm-start per hand

        All hands across all frames are packed into one (H, 778, 3) tensor and
        Adam optimizes (H, 3) translations in parallel. Each frame's depth_pc
        is subsampled (or replica-padded) to `batched_depth_subsample` points
        so cdist runs as one clean (H, 778, N_sub) op.

        Returns: list of lists — `result[f][h]` is the optimized translation
        for hand `h` of frame `f`, as (3,) np.float32.
        """
        device = self.device
        n_sub = int(self.opt_cfg.batched_depth_subsample)
        all_verts, all_cam_t, all_x0 = [], [], []
        all_K1, all_K2 = [], []
        frame_per_hand: List[int] = []
        hand_per_frame: List[List[int]] = [[] for _ in frames_data]
        depth_subs: List[np.ndarray] = []
        for fi, fd in enumerate(frames_data):
            for j in range(len(fd['all_verts'])):
                hand_per_frame[fi].append(len(all_verts))
                all_verts.append(fd['all_verts'][j])
                all_cam_t.append(fd['all_cam_t'][j])
                all_x0.append(fd['x0_list'][j])
                all_K1.append(fd['K1'])
                all_K2.append(fd['K2'])
                frame_per_hand.append(fi)
            pts = fd['depth_points']
            if pts.shape[0] >= n_sub:
                idx = np.random.RandomState(fi).choice(pts.shape[0], n_sub, replace=False)
            else:
                idx = np.random.RandomState(fi).choice(pts.shape[0], n_sub, replace=True)
            depth_subs.append(pts[idx])

        H = len(all_verts)
        if H == 0:
            return [[] for _ in frames_data]

        verts = torch.from_numpy(np.stack(all_verts)).to(device=device, dtype=torch.float32)         # (H, N, 3)
        cam_t = torch.from_numpy(np.stack(all_cam_t)).to(device=device, dtype=torch.float32)         # (H, 3)
        x0 = torch.from_numpy(np.stack(all_x0)).to(device=device, dtype=torch.float32)               # (H, 3)
        K1 = torch.from_numpy(np.stack(all_K1)).to(device=device, dtype=torch.float32)               # (H, 3, 3)
        K2 = torch.from_numpy(np.stack(all_K2)).to(device=device, dtype=torch.float32)               # (H, 3, 3)
        depth = torch.from_numpy(np.stack(depth_subs)).to(device=device, dtype=torch.float32)        # (F, n_sub, 3)
        frame_idx = torch.tensor(frame_per_hand, device=device, dtype=torch.long)                    # (H,)
        depth_per_hand = depth[frame_idx]                                                            # (H, n_sub, 3)

        V1 = verts + cam_t.unsqueeze(1)
        proj1 = torch.einsum('hij,hnj->hni', K1, V1)
        proj1_xy = proj1[..., :2] / proj1[..., 2:3].clamp(min=1e-6)

        t = x0.clone().detach().requires_grad_(True)
        optim = torch.optim.Adam([t], lr=self.opt_cfg.torch_lr)
        max_steps = int(self.opt_cfg.torch_steps)
        min_steps = int(self.opt_cfg.torch_min_steps)
        tol = float(self.opt_cfg.torch_tol)
        patience = 5
        best_loss = float("inf")
        best_t = t.detach().clone()
        no_improve = 0
        for step in range(max_steps):
            optim.zero_grad(set_to_none=True)
            V2 = verts + t.unsqueeze(1)
            proj2 = torch.einsum('hij,hnj->hni', K2, V2)
            proj2_xy = proj2[..., :2] / proj2[..., 2:3].clamp(min=1e-6)
            err_2d = ((proj1_xy - proj2_xy) ** 2).mean(dim=(1, 2))
            dists = torch.cdist(V2, depth_per_hand)
            nn = dists.min(dim=2).values
            err_3d = nn.mean(dim=1)
            loss = (err_2d + weight_3d * err_3d).sum()
            loss.backward()
            optim.step()
            if tol > 0.0:
                cur = loss.item()
                if cur < best_loss * (1.0 - tol):
                    best_loss = cur
                    best_t = t.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1
                    if step >= min_steps and no_improve >= patience:
                        break

        final_t = best_t if tol > 0.0 else t.detach()
        out_cpu = final_t.cpu().numpy().astype(np.float32)
        return [[out_cpu[h] for h in hand_idxs] for hand_idxs in hand_per_frame]

    def _hand_mask_fast(self, all_verts, all_cam_t, all_right, K1, img_shape):
        """Cheap replacement for the pyrender alpha pass.

        For each hand: project (verts + cam_t) through K1, take the 2D convex
        hull, fill it. The result is a (H, W) uint8 mask covering the hand
        region in image space — same use as `cam_view[:,:,3] > 0` in the
        pyrender path. Looser than the rendered alpha (ignores z-buffering and
        per-triangle silhouette edges) but the downstream `erode + invert + use
        as exclusion mask for depth_pc` is robust to that looseness.
        """
        H, W = img_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for verts, cam_t, is_right in zip(all_verts, all_cam_t, all_right):
            V = verts.copy()
            # Re-flip the x-axis side flag the same way the pyrender path does
            # (the renderer's `is_right` arg flips x internally; we mirror that).
            V[:, 0] = (2 * int(is_right) - 1) * V[:, 0]
            Vw = V + cam_t
            proj = Vw @ K1.T
            z = np.maximum(proj[:, 2:3], 1e-6)
            uv = proj[:, :2] / z
            uv = np.clip(uv, [0, 0], [W - 1, H - 1]).astype(np.int32)
            if uv.shape[0] >= 3:
                hull = cv2.convexHull(uv)
                cv2.fillConvexPoly(mask, hull, 1)
        return mask

    def _minimize_translation_torch(self, verts_list, cam_t_list, K1, K2, depth_points, x0_list,
                                     weight_3d: float):
        """GPU port of scipy minimize → batched torch Adam over both hands in-frame.

        Same objective as obj_function above:
            err_2d = ((K1 @ (verts + cam_t)) - (K2 @ (verts + x))).xy mse
            err_3d = mean nearest-neighbor distance from (verts + x) to depth_points
            total  = err_2d + weight_3d * err_3d

        Returns: list of (3,) np.float32 arrays — optimized translation per hand,
        in the order of `verts_list`.
        """
        device = self.device
        B = len(verts_list)
        if B == 0:
            return []
        # Stack hand inputs along a batch dim. Verts are always (778, 3); cam_t / x0 are (3,).
        verts = torch.from_numpy(np.stack(verts_list)).to(device=device, dtype=torch.float32)        # (B, N, 3)
        cam_t = torch.from_numpy(np.stack(cam_t_list)).to(device=device, dtype=torch.float32)        # (B, 3)
        x0 = torch.from_numpy(np.stack(x0_list)).to(device=device, dtype=torch.float32)              # (B, 3)
        K1_t = torch.from_numpy(np.asarray(K1)).to(device=device, dtype=torch.float32)               # (3, 3)
        K2_t = torch.from_numpy(np.asarray(K2)).to(device=device, dtype=torch.float32)               # (3, 3)
        depth_t = torch.from_numpy(np.ascontiguousarray(depth_points)).to(device=device, dtype=torch.float32)  # (M, 3)

        # Project V1 = verts + cam_t under K1. This term is constant w.r.t. the
        # optimization variable, so cache it once.
        V1 = verts + cam_t.unsqueeze(1)                                                              # (B, N, 3)
        proj1 = torch.matmul(V1, K1_t.T)                                                             # (B, N, 3)
        proj1_xy = proj1[..., :2] / proj1[..., 2:3].clamp(min=1e-6)                                  # (B, N, 2)

        # Make depth tensor broadcastable to (B, M, 3) without copying memory.
        depth_b = depth_t.unsqueeze(0).expand(B, -1, -1).contiguous()                                # (B, M, 3)

        # Track the best translation seen — Adam isn't monotonic near convergence
        # (NN target can flip between iters), so the final-step `t` may be worse
        # than an earlier one. We snapshot whenever loss improves and return the
        # snapshot, which dominates the "always return last step" alternative on
        # both speed (early-stop is robust) and quality (no oscillation tax).
        #
        # Pre-allocate the optimizer parameter tensor once per (B, device) and
        # reuse across frames. Adam state (moments) gets reset to zero each
        # frame because each frame is a *different* optimization problem, but
        # the tensor + optimizer object survive — saves the per-frame allocator
        # churn that was costing ~10-15 ms.
        if (self._adam_t is None or self._adam_t.shape[0] != B
                or self._adam_t.device != device):
            self._adam_t = torch.zeros(B, 3, device=device, requires_grad=True)
            self._adam_optim = torch.optim.Adam([self._adam_t], lr=self.opt_cfg.torch_lr)
        with torch.no_grad():
            self._adam_t.copy_(x0)
        # Reset Adam moments to zero (fresh problem) without rebuilding the optim.
        for state in self._adam_optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    v.zero_()
        t = self._adam_t
        optim = self._adam_optim
        max_steps = int(self.opt_cfg.torch_steps)
        min_steps = int(self.opt_cfg.torch_min_steps)
        tol = float(self.opt_cfg.torch_tol)
        patience = 5
        best_loss = float("inf")
        best_t = t.detach().clone()
        no_improve = 0
        for step in range(max_steps):
            optim.zero_grad(set_to_none=True)
            V2 = verts + t.unsqueeze(1)                                                              # (B, N, 3)
            proj2 = torch.matmul(V2, K2_t.T)                                                         # (B, N, 3)
            proj2_xy = proj2[..., :2] / proj2[..., 2:3].clamp(min=1e-6)                              # (B, N, 2)
            err_2d = ((proj1_xy - proj2_xy) ** 2).mean(dim=(1, 2))                                   # (B,)
            # 3D nearest-neighbor distance via cdist + min. (B, N, M) is the dominant
            # working set; for ~50k depth points × 778 verts × 2 hands that's ~78M
            # elements per step — well within A5000 memory.
            dists = torch.cdist(V2, depth_b)                                                         # (B, N, M)
            nn = dists.min(dim=2).values                                                             # (B, N)
            err_3d = nn.mean(dim=1)                                                                  # (B,)
            loss = (err_2d + weight_3d * err_3d).sum()
            loss.backward()
            optim.step()
            # Best-loss patience early-stop. `.item()` forces a host sync but
            # optimizer.step already does, so marginal cost is a float copy.
            # Skip entirely when tol == 0.
            if tol > 0.0:
                cur = loss.item()
                if cur < best_loss * (1.0 - tol):
                    best_loss = cur
                    best_t = t.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1
                    if step >= min_steps and no_improve >= patience:
                        break
        # Use the best snapshot if early-stop logic ran; otherwise the final t.
        final_t = best_t if tol > 0.0 else t.detach()
        return [final_t[i].cpu().numpy().astype(np.float32) for i in range(B)]

    def _derive_hand_bboxes(self, det_out, img):
        """Run ViTPose on the detector's person bboxes, return (boxes, right) or (None, None)."""
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        if pred_bboxes.shape[0] == 0:
            return None, None
        vitposes_out = self.cpm.predict_pose(
            img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )
        bboxes, is_right = [], []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            for keyp, right_flag in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                            keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(right_flag)
        if not bboxes:
            return None, None
        return np.stack(bboxes), np.stack(is_right)

    def extract_info(self, img_path: str, opt_weight: float, save_mesh: bool = False, frame_idx: int = 0):
        # Get the root directory name (parent folder name)
        parent_dir = os.path.dirname(img_path)

        # Read the image
        img_cv2 = cv2.imread(img_path)

        # Detector cache: rerun ViTDet only every `detector_stride` frames; reuse
        # the cached output otherwise. Auto-redetect on cache miss / failed reuse.
        stride = max(1, int(self.opt_cfg.detector_stride))
        det_out, used_cache = self._get_detector_output(img_cv2, frame_idx, stride)

        img = img_cv2[:, :, ::-1].copy()

        boxes, right = self._derive_hand_bboxes(det_out, img)

        # Safety net: cached body bbox produced no usable hand keypoints. Force a
        # fresh detect on this frame (and update the cache).
        if boxes is None and used_cache:
            det_out, _ = self._run_detector(img_cv2, frame_idx)
            boxes, right = self._derive_hand_bboxes(det_out, img)

        # If still no bounding boxes found, return empty arrays
        if boxes is None:
            return np.array([]), np.array([])

        # Optionally save the meshes
        if save_mesh:
            output_path, pcd, out = self._save_meshes(img_cv2, boxes, right, img_path, parent_dir, opt_weight)
        else:
            output_path, pcd, out= None, None, None

        return ExtractorOutput(hand_boxes=boxes, is_right=right, output_path=output_path, pcd=pcd, out=out, opt_weight=opt_weight)
        
    def convert_output_to_numpy(self, out):
        """
        Convert the output dictionary with PyTorch tensors and additional data (bboxes, is_right) to NumPy arrays.
        
        Args:
            out (dict): The output dictionary containing model outputs, including 'pred_mano_params'.
        
        Returns:
            dict: The dictionary with all tensors and additional data converted to NumPy arrays.
        """
        # Convert tensors in the main output dictionary to NumPy arrays
        out_numpy = {k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in out.items()}
        
        # Specifically handle 'pred_mano_params' if it exists
        if 'pred_mano_params' in out_numpy:
            out_numpy['pred_mano_params'] = {
                k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in out_numpy['pred_mano_params'].items()
            }
        
        return out_numpy


    def save_point_cloud_as_ply(self, vertices, output_folder, filename, colors=None):
        """
        Save a point cloud (with optional RGB colors) as a PLY file using Open3D.

        Args:
            vertices (numpy.ndarray): Nx3 array of 3D points.
            output_folder (str): Directory to save the PLY file.
            filename (str): Name of the output PLY file (without extension).
            colors (numpy.ndarray, optional): Nx3 array of RGB colors (values in range [0, 255]).
                                            If None, saves the point cloud without colors.

        Returns:
            str: Full path to the saved PLY file.
            open3d.geometry.PointCloud: The Open3D PointCloud object created and saved.
        
        Raises:
            ValueError: If the number of color entries does not match the number of vertices.
        """
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Create an Open3D PointCloud object (lazy import — only this method needs it).
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # Add colors if provided
        if colors is not None:
            if colors.shape[0] != vertices.shape[0]:
                raise ValueError("The number of color entries must match the number of vertices.")
            # Normalize RGB values to [0, 1] range
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Construct the full file path for the PLY file
        output_path = os.path.join(output_folder, f"{filename}.ply")

        # Save the point cloud to a PLY file. Binary PLY is faster to write and
        # is transparent to all downstream readers (rfp uses trimesh.load_mesh,
        # which auto-detects format).
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

        return output_path, pcd


    def save_output_as_npz(self, out, bboxes, is_right, filepath):
        """Convert tensors to numpy on the main thread (HaMeR `out` holds GPU
        refs that aren't safe to release-into-a-thread), then dispatch the
        actual `np.savez_compressed` to the writer pool. Overlaps disk I/O
        with the next frame's GPU work."""
        out_numpy = self.convert_output_to_numpy(out)
        out_numpy['bboxes'] = np.stack(bboxes)
        out_numpy['right'] = np.stack(is_right)
        # Trim already-resolved futures so the list doesn't grow unbounded.
        self._pending_writes = [f for f in self._pending_writes if not f.done()]
        self._pending_writes.append(
            self._writer.submit(np.savez_compressed, filepath, **out_numpy)
        )

    def flush_writes(self):
        """Block until all pending NPZ/PLY writes complete. Call before exit."""
        for f in self._pending_writes:
            f.result()
        self._pending_writes.clear()

    def flush_chunk(self, opt_weight: float):
        """Drain the per-chunk buffer: run one batched torch Adam minimize over
        all frames currently buffered, then dispatch the NPZ + (gated) PLY
        writes per frame. No-op if the buffer is empty.

        Called by the main loop at chunk boundaries and once at end-of-run.
        """
        states = self._chunk_buffer
        if not states:
            return
        self._chunk_buffer = []

        translations_per_frame = self._minimize_translation_torch_batched(states, opt_weight)

        for state, translations in zip(states, translations_per_frame):
            # Update HaMeR `out` dict with the optimized translations and the
            # per-frame bookkeeping that `save_output_as_npz` expects.
            out_numpy = state['out_numpy']
            out_numpy['opt_translation'] = np.asarray(translations) if translations else np.zeros((0, 3), dtype=np.float32)
            # Update warm-start cache from the LAST hand of the LAST frame in
            # the chunk for each side. Coarser than per-frame warm-start, but
            # the alternative is sequencing within the chunk which kills the
            # batching gain.
            if self.opt_cfg.warm_start:
                for i, tr in enumerate(translations):
                    self._last_opt_translation[int(state['all_right'][i])] = tr

            # NPZ save (async via writer pool)
            out_numpy['bboxes'] = np.stack(state['boxes'])
            out_numpy['right'] = np.stack(state['right_flags'])
            self._pending_writes = [f for f in self._pending_writes if not f.done()]
            self._pending_writes.append(self._writer.submit(
                np.savez_compressed,
                f"{state['model_out_folder']}/{state['img_fn']}.npz",
                **out_numpy,
            ))

            # 3dhand PLYs (gated, sync — open3d is C++ but holds the GIL on
            # write_point_cloud and is fast enough at binary PLY)
            if self.opt_cfg.save_3dhand_pcd:
                for i in range(len(state['all_verts'])):
                    self.save_point_cloud_as_ply(
                        state['all_verts'][i] + translations[i],
                        state['_3dhand_out_folder'],
                        f"{state['img_fn']}_{int(state['all_right'][i])}",
                    )

            # Scene PLY (gated, sync — same path as per-frame)
            if self.opt_cfg.save_scene_pcd:
                from mesh_to_sdf.rgbd2pc import RGBD2PC as _RGBD2PC
                scene_pcd = _RGBD2PC(
                    state['depth'], state['intrinsic_matrix'],
                    rgb=state['img_cv2_rgb'], camera_pose=np.eye(4),
                    target_mask=None, threshold=10.0,
                )
                scene_pcd.save_point_cloud(
                    os.path.join(state['scene_out_folder'], f"{state['img_fn']}.ply"))

    def _save_meshes(self, img_cv2, boxes, right, img_path, parent_dir, opt_weight: float = 100.0):
        # Create the output folder with 'hamer/root_dir_name' suffix
        out_root_dir = f"../out/hamer"
        plots_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/extra_plots")) # for plots and objs
        model_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/model")) # for model output
        _3dhand_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/3dhand")) # for hand aligned with fetch cam
        scene_out_folder = os.path.normpath(os.path.join(parent_dir, f"{out_root_dir}/scene")) # scene point cloud
        os.makedirs(plots_out_folder, exist_ok=True)
        os.makedirs(model_out_folder, exist_ok=True)
        if self.opt_cfg.save_3dhand_pcd:
            os.makedirs(_3dhand_out_folder, exist_ok=True)
        if self.opt_cfg.save_scene_pcd:
            os.makedirs(scene_out_folder, exist_ok=True)

        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        # Initialized so the (output_path, pcd, out) return is valid even
        # when save_3dhand_pcd is off (the default processing-mode path).
        output_path = None
        pcd = None

        depth = load_depth_img(img_path)

        # Use fp16 autocast on CUDA when enabled; pass-through on CPU.
        use_fp16 = self.opt_cfg.fp16 and self.device.type == "cuda"
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=use_fp16
            ):
                out = self.model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            
            # self.model_cfg.EXTRA.FOCAL_LENGTH = 574
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                # Per-hand debug renders (regression + side view) are pure visualization;
                # gate behind opt_cfg.save_debug_renders since they trigger pyrender + disk I/O per hand per frame.
                if self.opt_cfg.save_debug_renders:
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    regression_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                    )

                    side_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                white_img,
                                mesh_base_color=LIGHT_BLUE,
                                scene_bg_color=(1, 1, 1),
                                side_view=True)

                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    cv2.imwrite(os.path.join(plots_out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            # The mask is used only as an *exclusion* mask for depth_pc — i.e.,
            # to drop depth points that fall on the hand's own projection so
            # the translation optimizer fits against the surrounding scene
            # (table / object) instead of against the hand surface itself.
            #
            # 'fast' (default): cv2 convex-hull on projected verts, ~5 ms/frame
            # 'pyrender'      : full RGBA hand render, ~370 ms/frame
            #
            # Compute scaled_focal_length once (same K1 the optimizer uses).
            scaled_focal_length_val = (self.model_cfg.EXTRA.FOCAL_LENGTH /
                                        self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()).item()
            K1 = np.array([[scaled_focal_length_val, 0, 320],
                            [0, scaled_focal_length_val, 240],
                            [0, 0, 1]]).astype(np.float32)

            cam_view = None
            if self.opt_cfg.mask_backend == "pyrender" or self.opt_cfg.save_debug_renders:
                cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t,
                                                               render_res=img_size[n], is_right=all_right,
                                                               **misc_args)

            if self.opt_cfg.save_debug_renders:
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                cv2.imwrite(os.path.join(plots_out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

            if self.opt_cfg.mask_backend == "pyrender":
                mask = (cam_view[:, :, 3] > 0).astype(np.uint8)
            else:
                mask = self._hand_mask_fast(all_verts, all_cam_t, all_right, K1, img_cv2.shape)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel)
            mask = 1 - mask
            cam_K_path = os.path.join(parent_dir, '..', 'cam_K.txt')
            intrinsic_matrix = self._cam_K_cache.get(cam_K_path)
            if intrinsic_matrix is None:
                intrinsic_matrix = np.loadtxt(cam_K_path)
                self._cam_K_cache[cam_K_path] = intrinsic_matrix

            # convert depth to point cloud
            depth_pc = RGBD2PC(depth, intrinsic_matrix, camera_pose=np.eye(4), target_mask=mask, threshold=10.0, use_kmeans=True)

            # solve new translation; reuse K1 computed above for the mask path.
            K = K1
            x0 = np.mean(depth_pc.points, axis=0)
            
            out['opt_translation'] = []

            # Per-hand warm-start vectors (one entry per hand). Order matches all_verts.
            x0_list = []
            for i in range(len(all_verts)):
                hand_side = int(all_right[i])  # 0 left, 1 right
                if self.opt_cfg.warm_start and hand_side in self._last_opt_translation:
                    x0_list.append(self._last_opt_translation[hand_side])
                else:
                    x0_list.append(x0)

            # Cross-frame batching: defer minimize+save until the chunk fills.
            # Pre-convert GPU tensors in `out` to numpy so we can release GPU
            # memory while waiting for the rest of the chunk.
            if int(self.opt_cfg.frame_batch_size) > 1:
                state = {
                    'out_numpy': self.convert_output_to_numpy(out),
                    'boxes': boxes,
                    'right_flags': right,
                    'all_verts': list(all_verts),
                    'all_cam_t': list(all_cam_t),
                    'all_right': list(all_right),
                    'K1': K1.astype(np.float32),
                    'K2': np.asarray(intrinsic_matrix, dtype=np.float32),
                    'depth_points': depth_pc.points.astype(np.float32, copy=True),
                    'x0_list': [np.asarray(v, dtype=np.float32) for v in x0_list],
                    'model_out_folder': model_out_folder,
                    '_3dhand_out_folder': _3dhand_out_folder,
                    'scene_out_folder': scene_out_folder,
                    'img_fn': img_fn,
                    # Only carry depth/img_cv2 forward if scene PLY save is enabled,
                    # otherwise we'd hold ~700 KB × K frames of pixel data needlessly.
                    'img_cv2_rgb': (img_cv2.astype(np.float32)[:, :, ::-1].copy()
                                     if self.opt_cfg.save_scene_pcd else None),
                    'depth': (depth.copy() if self.opt_cfg.save_scene_pcd else None),
                    'intrinsic_matrix': intrinsic_matrix,
                }
                self._chunk_buffer.append(state)
                return None, None, None

            if self.opt_cfg.minimize_backend == "torch":
                # GPU Adam loop, batched across both hands. Replaces the per-hand
                # scipy Nelder-Mead loop on CPU; cuts ~300+ ms/frame on this rig.
                translations = self._minimize_translation_torch(
                    verts_list=all_verts, cam_t_list=all_cam_t,
                    K1=K, K2=intrinsic_matrix, depth_points=depth_pc.points,
                    x0_list=x0_list, weight_3d=opt_weight,
                )
            else:
                opt_options = {
                    'xatol': self.opt_cfg.xatol,
                    'maxiter': self.opt_cfg.maxiter,
                    'disp': self.opt_cfg.disp,
                }
                translations = []
                for i in range(len(all_verts)):
                    res = minimize(obj_function, x0_list[i], method='nelder-mead',
                                   args=(all_verts[i], all_cam_t[i], K, intrinsic_matrix,
                                         depth_pc.kd_tree, opt_weight),
                                   options=opt_options)
                    translations.append(res.x)

            for i in range(len(all_verts)):
                hand_side = int(all_right[i])
                out['opt_translation'].append(translations[i])
                if self.opt_cfg.warm_start:
                    self._last_opt_translation[hand_side] = translations[i]

            out['opt_translation'] = np.asarray(out['opt_translation'])
        
            # save the model output
            self.save_output_as_npz(out, boxes, right, f"{model_out_folder}/{img_fn}.npz")

            # fig = plt.figure()
            # ax = fig.add_subplot(1, 3, 1)
            # plt.imshow(input_img)
            
            # # verify projection 1
            # vertices = all_verts[-1] + all_cam_t[-1]
            # print(K, vertices)
            # print(vertices.shape)
            # x2d = K @ vertices.T
            # x2d[0, :] /= x2d[2, :]
            # x2d[1, :] /= x2d[2, :]
            # plt.plot(x2d[0, :], x2d[1, :])
            # plt.title('projection using hamer camera')

            # ax = fig.add_subplot(1, 3, 2)
            # plt.imshow(input_img)

            # verify projection 2
            # vertices = all_verts[-1] + translation_new
            # x2d = intrinsic_matrix @ vertices.T
            # x2d[0, :] /= x2d[2, :]
            # x2d[1, :] /= x2d[2, :]
            # plt.plot(x2d[0, :], x2d[1, :])              
            # plt.title('projection using fetch camera')

            # ax = fig.add_subplot(1, 3, 3, projection='3d')
        
            # ax.scatter(depth_pc.points[:, 0], depth_pc.points[:, 1], depth_pc.points[:, 2], marker='o')
            # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', color='r')

            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()               

            # save rgbd scene pc — opt-in only.
            # Only consumed by scripts/tools/seq_viewer.py; rfp-grasp-transfer
            # never reads scene/*.ply. ASCII-PLY of ~300k points dominated wall-
            # clock at ~3 s/frame, so we gate this whole branch behind
            # opt_cfg.save_scene_pcd. When enabled, the writer uses binary PLY.
            if self.opt_cfg.save_scene_pcd:
                img_cv2 = img_cv2.astype(np.float32)[:, :, ::-1]  # Convert from BGR to RGB
                RT = np.eye(4)
                scene_pcd = RGBD2PC(depth, intrinsic_matrix, rgb=img_cv2, camera_pose=RT, target_mask=None, threshold=10.0)
                scene_pcd.save_point_cloud(os.path.join(scene_out_folder, f"{img_fn}.ply"))

            # save fetch cam aligned hamer hand mesh pc — opt-in, viz-only.
            # rfp consumes verts via model/*.npz (pred_vertices + opt_translation),
            # not via this PLY. The PLY is only for the viz tools (seq_viewer.py,
            # create_combined_ply.py, rfp's own --save_viz path).
            if self.opt_cfg.save_3dhand_pcd:
                for i in range(len(all_verts)):
                    output_path, pcd = self.save_point_cloud_as_ply(
                        all_verts[i] + out['opt_translation'][i],
                        _3dhand_out_folder, f"{img_fn}_{int(all_right[i])}")

        return output_path, pcd, out


if __name__ == "__main__":
    vlog.section("HaMeR — hand bbox + 3D mesh extraction")

    parser = argparse.ArgumentParser(description="Process images for hand mesh bounding box extraction.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--opt_weight", type=float, default=100.0, help="weight for hamer hand mesh optimization with depth")
    parser.add_argument("--opt_xatol", type=float, default=1e-4, help="Nelder-Mead xatol convergence tolerance (was 1e-8 hardcoded).")
    parser.add_argument("--opt_maxiter", type=int, default=30, help="Nelder-Mead max iterations.")
    parser.add_argument("--opt_disp", action="store_true", help="Print scipy minimize convergence info per call (slow).")
    parser.add_argument("--save_debug_renders", action="store_true", help="Save per-frame regression/side/all overlay PNGs (slow; off by default).")
    parser.add_argument("--save_viz", action="store_true",
                        help="Unified across vie scripts: save all visualization/debug "
                             "artifacts for this stage. Implies --save_debug_renders and "
                             "--save_scene_pcd.")
    parser.add_argument("--save_scene_pcd", action="store_true",
                        help="Write out/hamer/scene/*.ply per frame. Off by default — "
                             "rfp-grasp-transfer doesn't read it; only seq_viewer.py does. "
                             "Saves ~3 s/frame on this rig when off.")
    parser.add_argument("--save_3dhand_pcd", action="store_true",
                        help="Write out/hamer/3dhand/*.ply per hand per frame. Off by "
                             "default — rfp computes hand verts from model/*.npz "
                             "(pred_vertices + opt_translation); the PLY is only read "
                             "by seq_viewer / create_combined_ply / rfp's --save_viz "
                             "path. Saves ~45 ms/frame on this rig when off.")
    parser.add_argument("--detector_stride", type=int, default=5,
                        help="Run ViTDet body detection every K frames; reuse the cached "
                             "bbox in between (auto-redetect on cache miss). 1 = every frame.")
    parser.add_argument("--minimize_backend", type=str, default="torch", choices=["torch", "scipy"],
                        help="Translation refinement backend. 'torch' uses GPU Adam "
                             "batched across both hands (~5-10x faster); 'scipy' is the "
                             "original Nelder-Mead, kept for parity testing.")
    parser.add_argument("--torch_steps", type=int, default=50,
                        help="Adam steps for the torch minimize backend.")
    parser.add_argument("--torch_lr", type=float, default=5e-3,
                        help="Adam learning rate for the torch minimize backend.")
    parser.add_argument("--torch_min_steps", type=int, default=15,
                        help="Minimum Adam steps before early-stop is allowed.")
    parser.add_argument("--torch_tol", type=float, default=1e-3,
                        help="Relative-loss-change tolerance for Adam early-stop. "
                             "0 disables early-stop (always run torch_steps iters).")
    parser.add_argument("--frame_batch_size", type=int, default=1,
                        help="Cross-frame GPU batching: when > 1, buffer K frames "
                             "of per-frame state and run ONE batched torch Adam over "
                             "all hands × K frames at once. Saves ~50-80 ms/frame on "
                             "this rig at K=4. 1 = per-frame (unchanged). Quality "
                             "tradeoff: within-chunk warm-start is coarser.")
    parser.add_argument("--batched_depth_subsample", type=int, default=5000,
                        help="Per-frame depth-pc point count for the batched-minimize "
                             "path (frame_batch_size > 1). Each frame is subsampled "
                             "to this many points. Default 5000 — sweet spot from "
                             "micro-bench.")
    parser.add_argument("--mask_backend", type=str, default="fast", choices=["fast", "pyrender"],
                        help="Hand-region mask used to exclude depth points on the hand "
                             "from the depth_pc target. 'fast' = projected-vert convex hull "
                             "(~5 ms/frame); 'pyrender' = upstream RGBA render (~370 ms/frame).")
    parser.add_argument("--no_warm_start", action="store_true", help="Disable warm-starting Nelder-Mead from prior frame's translation (per hand side).")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 autocast on the HaMeR transformer forward pass (default on, CUDA-only).")
    parser.add_argument("--parallel_load", action="store_true",
                        help="Load ViTDet/ViTPose/HaMeR concurrently on 3 threads. Off "
                             "by default — on this rig the loads are GIL-bound and "
                             "threading actually slows the phase. May help on machines "
                             "where checkpoint reads are more I/O-dominated.")
    parser.add_argument("--no_model_cache", action="store_true",
                        help="Disable the on-disk pickle cache for detector + cpm at "
                             "~/.cache/hamer/. First run builds fresh and writes cache "
                             "(~5 GB); subsequent runs load from cache instead of "
                             "rebuilding ViTDet+ViTPose (~18 s saved). HaMeR is always "
                             "loaded from its ckpt (its LightningModule holds a ctypes "
                             "pointer that can't be pickled).")
    parser.add_argument("--body_detector", type=str, default="vitdet", choices=["vitdet", "regnety"],
                        help="Body detector for hand-bbox proposals. 'vitdet' is the upstream default "
                             "(ViTDet-Huge, ~2.5GB VRAM); 'regnety' is much smaller (~250MB) and "
                             "useful on GPUs <12GB VRAM where the full HaMeR + ViTPose + ViTDet stack OOMs.")
    args = parser.parse_args()

    input_dir = args.input_dir
    opt_weight = args.opt_weight

    # Two modes — `processing` (default, what rfp consumes) vs `viz` (--save_viz).
    # Processing mode produces only model/*.npz and 3dhand/*.ply, the artifacts
    # rfp-grasp-transfer reads. Pyrender is never invoked. Viz mode adds the
    # debug renders + scene point clouds and switches the mask backend back to
    # the upstream pyrender path so the rendered overlays match the mask used.
    if args.save_viz and args.mask_backend == "fast":
        # User asked for viz but didn't override mask: use pyrender for fidelity.
        args.mask_backend = "pyrender"

    opt_cfg = OptConfig(
        xatol=args.opt_xatol,
        maxiter=args.opt_maxiter,
        disp=args.opt_disp,
        save_debug_renders=args.save_debug_renders or args.save_viz,
        warm_start=not args.no_warm_start,
        fp16=not args.no_fp16,
        detector_stride=args.detector_stride,
        save_scene_pcd=args.save_scene_pcd or args.save_viz,
        save_3dhand_pcd=args.save_3dhand_pcd or args.save_viz,
        minimize_backend=args.minimize_backend,
        torch_steps=args.torch_steps,
        torch_lr=args.torch_lr,
        torch_min_steps=args.torch_min_steps,
        torch_tol=args.torch_tol,
        mask_backend=args.mask_backend,
        frame_batch_size=args.frame_batch_size,
        batched_depth_subsample=args.batched_depth_subsample,
    )
    is_viz_mode = (args.save_viz or args.save_debug_renders or
                   args.save_scene_pcd or args.save_3dhand_pcd)

    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        vlog.error(f"Input directory does not exist: {input_dir}")
        raise SystemExit(2)

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png'))]
    if not image_files:
        vlog.error(f"No image files found in {input_dir}")
        raise SystemExit(2)

    image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))

    vlog.section("Configuration")
    mode_label = "[bold yellow]viz[/]" if is_viz_mode else "[bold green]processing[/] (rfp-ready)"
    vlog.note(f"mode              : {mode_label}")
    vlog.note(f"input dir         : {input_dir}")
    vlog.note(f"frames            : {len(image_files)}")
    vlog.note(f"body detector     : {args.body_detector}  (stride={opt_cfg.detector_stride})")
    vlog.note(f"opt weight        : {opt_weight}")
    vlog.note(f"opt xatol/maxiter : {opt_cfg.xatol} / {opt_cfg.maxiter}  (warm_start={opt_cfg.warm_start})")
    vlog.note(f"minimize backend  : {opt_cfg.minimize_backend}  "
              f"(steps={opt_cfg.torch_steps} max, min={opt_cfg.torch_min_steps}, "
              f"tol={opt_cfg.torch_tol}, lr={opt_cfg.torch_lr})")
    vlog.note(f"frame_batch_size  : {opt_cfg.frame_batch_size}"
              + (f"  (depth subsample to {opt_cfg.batched_depth_subsample} pts)"
                 if opt_cfg.frame_batch_size > 1 else ""))
    vlog.note(f"mask backend      : {opt_cfg.mask_backend}")
    vlog.note(f"fp16              : {opt_cfg.fp16}")
    vlog.note(f"save_scene_pcd    : {opt_cfg.save_scene_pcd}")
    vlog.note(f"save_3dhand_pcd   : {opt_cfg.save_3dhand_pcd}")
    vlog.note(f"save_debug_renders: {opt_cfg.save_debug_renders}")

    vlog.section("Loading models")
    # Per-step spinners now live inside HandInfoExtractor.__init__ so the user
    # sees individual progress for "Importing ML stack", "Loading HaMeR",
    # "Loading ViTDet", "Loading ViTPose", etc. — instead of one combined
    # ~45 s "Loading HaMeR + ViTDet + ViTPose" spinner.
    hand_info_extractor = HandInfoExtractor(
        opt_cfg=opt_cfg, body_detector=args.body_detector,
        parallel_load=args.parallel_load,
        use_model_cache=not args.no_model_cache,
    )

    vlog.section("Tracking")
    import time as _time
    frame_times: List[float] = []
    n_skipped = 0
    t_total = _time.time()
    chunk_k = int(opt_cfg.frame_batch_size)
    with vlog.progress("Extracting hand bboxes + 3D meshes", total=len(image_files)) as (prog, task):
        for frame_idx, img_file in enumerate(image_files):
            img_path = os.path.join(input_dir, img_file)
            t_frame = _time.time()
            try:
                hand_info_extractor.extract_info(img_path, opt_weight=opt_weight,
                                                  save_mesh=True, frame_idx=frame_idx)
                frame_times.append(_time.time() - t_frame)
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 3:
                    import traceback
                    vlog.warn(f"skip {img_file}: {e}")
                    traceback.print_exc()
            prog.update(task, advance=1)
            # Chunked-batching path: flush when the buffer reaches K frames.
            # When chunk_k == 1 the buffer is never populated (extract_info
            # runs minimize inline), so this is a no-op.
            if chunk_k > 1 and len(hand_info_extractor._chunk_buffer) >= chunk_k:
                hand_info_extractor.flush_chunk(opt_weight)
        # Drain any remaining frames in the last partial chunk.
        if chunk_k > 1:
            hand_info_extractor.flush_chunk(opt_weight)
    # Drain the async NPZ writer pool — the loop dispatches saves but doesn't
    # block on them. flush_writes() ensures the disk reflects every frame
    # before we print "Done" and the writer thread tears down.
    hand_info_extractor.flush_writes()
    total_dt = _time.time() - t_total

    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        det_runs = hand_info_extractor.detector_runs
        det_skips = hand_info_extractor.detector_skips
        det_total = max(det_runs + det_skips, 1)
        vlog.summary({
            "frames processed": f"{len(frame_times)} / {len(image_files)} ({n_skipped} skipped)",
            "avg ms/frame":     f"{avg * 1000:.1f}",
            "fps":              vlog.fmt_rate(len(frame_times), sum(frame_times)),
            "total wall":       vlog.fmt_duration(total_dt),
            "detector runs":    f"{det_runs} / {det_total}  (stride={opt_cfg.detector_stride}, "
                                f"reuse={100 * det_skips / det_total:.0f}%)",
            "output dir":       os.path.normpath(os.path.join(input_dir, "..", "out", "hamer")),
        }, title="HaMeR")
        # Keep the bench-friendly grep line so scripts/bench_vie.sh can still parse.
        print(f"[hamer] processed {len(frame_times)} frames | avg {avg*1000:.1f} ms/frame | total {sum(frame_times):.1f}s")
        vlog.success("Done.")
    else:
        vlog.error("No frames processed successfully.")
        raise SystemExit(1)
