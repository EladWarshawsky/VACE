#!/bin/bash

set -euo pipefail

# ===================================================================================
# VACE with FreeLong++: End-to-End Long Video Generation with Depth Control
# ===================================================================================
# This script automates the entire process of generating a long video conditioned
# on a depth map sequence derived from an input video.
#
# It performs two main steps:
# 1. Preprocessing: Takes a source video and generates a corresponding depth map video.
# 2. Inference: Uses the depth video as a control signal (`src_video`) to guide the
#    FreeLong++ enabled VACE model, generating a new, longer video based on a text prompt.
# ===================================================================================

# --- Configuration ---
# You can change these variables to test with different inputs.
INPUT_VIDEO="./without_first_frame.mp4"
TEXT_PROMPT="some kids speak in the desert, the girl in the foreground speaking first, then her friend speaks disgruntled in the back, putting her hands on her hips, in response the girl in the front keeps talking"
OUTPUT_DIR="results/freelong_depth_demo_$(date +%s)"
FRAME_NUM=216 # 4x the native length of Wan2.1 (81 frames)

# --- FreeLong++ Configuration ---
# This JSON string configures the behavior of the FreeLongWanAttentionBlock.
# - native_video_length: The original training length of the base model.
# - long_video_scaling_factors: A list of window size multipliers for the multi-branch attention.
#   For 4x generation, [1, 2, 4] is a good choice.
# - sparse_key_frame_ratio: The fraction of frames to use as keys/values in the global attention branch.
FREELONG_CONFIG_JSON='{"native_video_length": 81, "long_video_scaling_factors": [1, 2, 4], "sparse_key_frame_ratio": 0.5}'

echo "--- VACE & FreeLong++ Long Video Generation ---"
echo "Input Video: $INPUT_VIDEO"
echo "Prompt: $TEXT_PROMPT"
echo "Output Directory: $OUTPUT_DIR"
echo "-------------------------------------------------"
echo ""

# Ensure dependencies are available in the base environment using uv for installs
# Prefer user's conda base Python if available
if [ -x "/Users/user/miniconda3/bin/python" ]; then
  PY_BIN="/Users/user/miniconda3/bin/python"
else
  PY_BIN=$(command -v python)
fi
echo "INFO: Python binary: ${PY_BIN}"
${PY_BIN} -V || true
${PY_BIN} -c 'import sys; print("Python executable:", sys.executable)' || true
echo "INFO: Ensuring Python dependencies (easydict, numpy) with uv in base env..."
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PY_BIN}" -q easydict numpy || true
else
  echo "WARNING: 'uv' not found; proceeding without auto-install."
fi

# Ensure local Wan2.1 package is importable
export PYTHONPATH="$(pwd)/Wan2.1:${PYTHONPATH:-}"

# Quick import check with target interpreter
"${PY_BIN}" - <<'PY'
try:
    import easydict, numpy
    print('INFO: Verified imports: easydict, numpy')
except Exception as e:
    print('WARNING: Import check failed:', e)
PY

# --- Step 1: Preprocessing video to generate depth map sequence ---
echo "INFO: Running preprocessing to generate depth map video..."

# Execute the preprocessing script with uv. We capture its standard output to find the path
# to the generated depth video, which will be used as input for the next step.
PREPROCESS_OUTPUT=$("${PY_BIN}" vace/vace_preproccess.py --task depth --video "$INPUT_VIDEO")

# Extract the 'src_video' path from the output log using grep and sed.
SRC_VIDEO_PATH=$(echo "$PREPROCESS_OUTPUT" | grep 'Save frames result to' | sed -n 's/Save frames result to //p')

if [ -z "$SRC_VIDEO_PATH" ]; then
    echo "ERROR: Could not find the path to the preprocessed depth video. Preprocessing might have failed. Aborting."
    exit 1
fi

echo "SUCCESS: Depth video successfully generated at: $SRC_VIDEO_PATH"
echo ""

# --- Step 2: Running VACE with FreeLong++ for Long Video Generation ---
echo "INFO: Running main inference with FreeLong++ enabled..."

# Execute the main inference script via uv.
# --src_video is now the path to our depth map video.
# --use_freelong enables the new attention block and SpecMix noise.
# --frame_num is set to a long video length.
"${PY_BIN}" vace/vace_wan_inference.py \
    --ckpt_dir 'models/Wan2.1-VACE-1.3B/' \
    --model_name 'vace-1.3B' \
    --src_video "$SRC_VIDEO_PATH" \
    --prompt "$TEXT_PROMPT" \
    --save_dir "$OUTPUT_DIR" \
    --frame_num $FRAME_NUM \
    --use_freelong \
    --freelong_config_json "$FREELONG_CONFIG_JSON"

echo ""
echo "--- Long video generation complete! ---"
echo "Final output saved in: $OUTPUT_DIR"
