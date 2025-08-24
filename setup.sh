#!/bin/bash
set -e  # exit if any command fails

pip install uv 

uv venv vacer --python=3.13.2

source vacer/bin/activate

# Install deps
uv pip install -r requirements.txt

mkdir -p models/VACE-Annotators/depth

# Download annotator model
wget -O models/VACE-Annotators/depth/dpt_hybrid-midas-501f0c75.pt \
  https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt

mkdir -p models/Wan2.1-VACE-1.3B

# Download Wan2.1-VACE-1.3B model
hf download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir models/Wan2.1-VACE-1.3B

cd ~/VACE/Wan2.1
uv pip install -e . --no-deps