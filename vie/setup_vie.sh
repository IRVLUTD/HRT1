#!/bin/bash

#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

# Purpose: vie eco-system setup. Bakes in every workaround we hit while standing
# this up on a fresh Blackwell laptop:
#  - pin transformers downward so old GroundingDINO doesn't error on get_head_mask
#  - pin setuptools<70 so legacy mmcv setup.py (which imports pkg_resources) builds
#  - install mmcv 1.5.0 explicitly with --no-build-isolation (HaMeR's setup.py pins
#    1.3.9 but mmpose 0.24 only accepts <=1.5.0; build env needs the env's setuptools)
#  - patch GroundingDINO's ms_deform_attn.py to fall back to the pure-PyTorch
#    implementation when the _C CUDA extension isn't compiled (rebuilding the C
#    op needs a full CUDA toolchain match with torch's cuda version)
#  - re-pin numpy<2 so matplotlib + downstream c-extensions don't break
#  - on Blackwell (sm_120) GPUs torch 2.3.1+cu118 lacks kernels; bump if needed

set -euo pipefail

ROOT_DIR=$PWD

echo "==> Step 1: torch + base deps"
conda install -y pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

echo "==> Step 2: pip install requirements.txt"
pip install -r requirements.txt

echo "==> Step 3: robokit setup.py (pulls GroundingDINO, MobileSAM, SAM2 + checkpoints)"
python setup.py install
pip install --upgrade --force-reinstall hydra-core   # samv2 needs a recent hydra

# GroundingDINO from the pinned commit ships csrc but the pip install of it
# typically does NOT build _C unless a matching CUDA toolchain is on PATH. If
# _C is missing the model errors at first call. Apply a runtime fallback so
# ms_deform_attn uses the pure-PyTorch path when _C isn't available.
echo "==> Step 3b: patch GroundingDINO ms_deform_attn for missing-_C fallback"
python - <<'PY'
import re, importlib.util, pathlib
spec = importlib.util.find_spec("groundingdino.models.GroundingDINO.ms_deform_attn")
if spec is None or spec.origin is None:
    print("  [skip] groundingdino not importable; nothing to patch")
else:
    p = pathlib.Path(spec.origin)
    src = p.read_text()
    if "_HAS_C" not in src:
        src = src.replace(
            "try:\n    from groundingdino import _C\nexcept:\n"
            '    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")\n',
            "try:\n    from groundingdino import _C\n    _HAS_C = True\n"
            "except Exception:\n"
            '    warnings.warn("groundingdino._C not available — falling back to pure-PyTorch ms_deform_attn.")\n'
            "    _C = None\n    _HAS_C = False\n",
        )
        src = src.replace(
            "if torch.cuda.is_available() and value.is_cuda:",
            "if torch.cuda.is_available() and value.is_cuda and _HAS_C:",
        )
        p.write_text(src)
        print(f"  patched {p}")
    else:
        print("  [skip] already patched")
PY

# transformers >=5 removed BertModel.get_head_mask which the pinned old
# GroundingDINO needs. Pin downward.
echo "==> Step 3c: pin transformers==4.47.1 (old GDINO needs get_head_mask)"
pip install "transformers==4.47.1"

echo "==> Step 4: clone + install HaMeR (only if vie/hamer is missing)"
if [ ! -d "$ROOT_DIR/hamer" ]; then
    git clone --recursive https://github.com/geopavlakos/hamer.git
    cd hamer
    git checkout df533a2d04b9e2ece7cf9d6cbc6982e140210517
    cd "$ROOT_DIR"
fi

# mmcv 1.x's setup.py imports pkg_resources, which newer setuptools dropped.
# Pin setuptools<70 so the build succeeds.
echo "==> Step 4b: pin setuptools<70 then install mmcv 1.5.0 + ViTPose"
pip install "setuptools<70"
pip install "mmcv==1.5.0" --no-build-isolation
pip install -v -e "$ROOT_DIR/hamer/third-party/ViTPose"

echo "==> Step 4c: install hamer as editable (--no-deps so we don't fight its strict mmcv pin)"
pip install --no-deps -e "$ROOT_DIR/hamer"

# HaMeR's editable install can drag in numpy>=2 via fresh wheels; pin back.
echo "==> Step 4d: pin numpy<2 (matplotlib + c-extensions need 1.x)"
pip install "numpy<2"

echo "==> Step 5: fetch HaMeR demo data + checkpoints (~couple GB)"
cd "$ROOT_DIR/hamer"
bash fetch_demo_data.sh
cd "$ROOT_DIR"

echo
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo
echo "MANO models (REQUIRED, license-gated, can't be scripted):"
echo "  1. Register at https://mano.is.tue.mpg.de/  (free)"
echo "  2. Download mano_v1_2.zip and extract"
echo "  3. Copy MANO_RIGHT.pkl and MANO_LEFT.pkl to:"
echo "     $ROOT_DIR/hamer/_DATA/data/mano/"
echo
echo "GPU note: if you're on a Blackwell GPU (RTX 50-series / sm_120) and the"
echo "default conda-installed torch lacks sm_120 kernels, bump it with:"
echo '  pip install --upgrade torch torchvision \\'
echo '      --index-url https://download.pytorch.org/whl/cu130'
echo
