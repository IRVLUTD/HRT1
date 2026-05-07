#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------
# Benchmark wrapper for the vie pipeline. Runs each module on a task dir with timing instrumentation
# and prints a unified summary.
#
# Usage:
#   ./bench_vie.sh <task_data_root> [text_prompt]
#
#   task_data_root : path containing rgb/, depth/, cam_K.txt, pose/
#   text_prompt    : optional GDINO+SAMv2 prompt (default: read from
#                    out/bundlesdf/demonstration/obj_prompt_mapper.json if present,
#                    else fall back to "object")
#
# Steps run (skips any whose inputs aren't present):
#   1. gdino+samv2  (run_gdino_samv2.py)
#   2. hamer        (extract_hand_bboxes_and_meshes.py)
#   3. grasp-transfer (rfp-grasp-transfer/transfer_from_hamer.py)
#   4. bundlesdf    (only if Docker is available; otherwise skipped with a note)
#
# Each module prints its own [module] timing line. This script tees stdout to
# bench_<timestamp>.log under task_data_root for later diffing across runs.
#
# A/B usage:
#   git checkout main && ./bench_vie.sh /path/to/task ...
#   git checkout jishnu/fasten-vie && ./bench_vie.sh /path/to/task ...
#   diff bench_<ts1>.log bench_<ts2>.log
# ----------------------------------------------------------------------------------------------------

set -uo pipefail

TASK_ROOT="${1:-}"
TEXT_PROMPT="${2:-}"

if [[ -z "$TASK_ROOT" ]]; then
    echo "usage: $0 <task_data_root> [text_prompt]" >&2
    exit 2
fi
if [[ ! -d "$TASK_ROOT" ]]; then
    echo "task root not found: $TASK_ROOT" >&2
    exit 2
fi

VIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$TASK_ROOT/bench_${TS}.log"

# Resolve prompt if not given.
if [[ -z "$TEXT_PROMPT" ]]; then
    PROMPT_JSON="$TASK_ROOT/out/bundlesdf/demonstration/obj_prompt_mapper.json"
    if [[ -f "$PROMPT_JSON" ]]; then
        TEXT_PROMPT="$(python -c "import json,sys; d=json.load(open('$PROMPT_JSON')); print(next(iter(d.values())))" 2>/dev/null || echo "object")"
    else
        TEXT_PROMPT="object"
    fi
fi

# Activate the gsam2 env if it exists (modules 1,3,4 use it; hamer uses robokit).
have_conda=0
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && have_conda=1
fi

activate() {
    [[ "$have_conda" -eq 1 ]] || return 0
    conda activate "$1" 2>/dev/null || echo "[bench] (warning) could not activate conda env: $1" >&2
}

section() {
    printf '\n========================================\n[bench] %s\n========================================\n' "$1"
}

run_step() {
    local name="$1"; shift
    section "$name"
    local t0
    t0=$(date +%s.%N)
    "$@"
    local rc=$?
    local t1
    t1=$(date +%s.%N)
    local dt
    dt=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
    if [[ $rc -ne 0 ]]; then
        printf '[bench] %s FAILED (rc=%d) after %ss\n' "$name" "$rc" "$dt"
    else
        printf '[bench] %s OK in %ss\n' "$name" "$dt"
    fi
    return 0  # never abort the whole bench on one module's failure
}

{
    echo "[bench] task_root: $TASK_ROOT"
    echo "[bench] text_prompt: $TEXT_PROMPT"
    echo "[bench] vie_root: $VIE_ROOT"
    echo "[bench] commit: $(git -C "$VIE_ROOT/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "[bench] branch: $(git -C "$VIE_ROOT/.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "[bench] timestamp: $TS"

    # 1. gdino + samv2 ------------------------------------------------------
    if [[ -d "$TASK_ROOT/rgb" ]]; then
        activate gsam2-3.10
        run_step "1/4 gdino+samv2" \
            python "$VIE_ROOT/run_gdino_samv2.py" \
                --input_dir "$TASK_ROOT/rgb" \
                --text_prompt "$TEXT_PROMPT" \
                --save_interval=1
    else
        echo "[bench] skipping gdino+samv2 — no $TASK_ROOT/rgb"
    fi

    # 2. hamer --------------------------------------------------------------
    if [[ -d "$TASK_ROOT/rgb" && -d "$TASK_ROOT/depth" ]]; then
        activate robokit-py3.10
        run_step "2/4 hamer" \
            python "$VIE_ROOT/hamer/extract_hand_bboxes_and_meshes.py" \
                --opt_weight 100.0 \
                --input_dir "$TASK_ROOT/rgb"
    else
        echo "[bench] skipping hamer — need $TASK_ROOT/rgb and $TASK_ROOT/depth"
    fi

    # 3. rfp-grasp-transfer -------------------------------------------------
    HAMER_MODELS="$VIE_ROOT/hamer/_DATA/data/mano/mano_v1_2/models"
    if [[ -d "$TASK_ROOT/out/hamer/model" && -d "$HAMER_MODELS" ]]; then
        activate robokit-py3.10
        run_step "3/4 rfp-grasp-transfer" \
            python "$VIE_ROOT/rfp-grasp-transfer/transfer_from_hamer.py" \
                --mano_model_dir "$HAMER_MODELS" \
                --target_gripper fetch_gripper \
                --input_dir "$TASK_ROOT"
    else
        echo "[bench] skipping rfp-grasp-transfer — need hamer model output and $HAMER_MODELS"
    fi

    # 4. bundlesdf ----------------------------------------------------------
    if command -v docker >/dev/null 2>&1 && [[ -d "$VIE_ROOT/BundleSDF" ]]; then
        DEMO_FRAMES="${BENCH_DEMO_FRAMES:-15}"
        ROLLOUT_FRAMES="${BENCH_ROLLOUT_FRAMES:-5}"
        run_step "4/4 bundlesdf" \
            "$VIE_ROOT/run_bundlesdf.sh" "$TASK_ROOT" "$DEMO_FRAMES" "$ROLLOUT_FRAMES"
    else
        echo "[bench] skipping bundlesdf — needs docker on PATH"
    fi

    echo
    echo "[bench] done. Log: $LOG"
} 2>&1 | tee "$LOG"
