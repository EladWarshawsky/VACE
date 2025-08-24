#!/bin/bash
set -e  # exit if any command fails

# Install deps
pip install -r requirements.txt

# Download annotator model
wget -O models/VACE-Annotators/depth/dpt_hybrid-midas-501f0c75.pt \
  https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt

# Download Wan2.1-VACE-1.3B model
hf download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir models/Wan2.1-VACE-1.3B
