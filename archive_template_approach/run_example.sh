#!/bin/bash
# Example: Run collembola detection on sample image
# This script demonstrates the recommended usage

set -e  # Exit on error

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate collembola

# Create output directory
mkdir -p out

# Run detection with optimized settings
python sam_templates.py \
    "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --downscale-max-side 2048 \
    --scale-factors 1.0 \
    --ncc-threshold 0.65 \
    --max-prompts 100 \
    --output out/measurements.csv \
    --json out/summary.json \
    --save-overlay out/overlay.png \
    --save-masks-dir out/masks \
    --auto-download \
    --allow-large-image

echo ""
echo "================================================"
echo "Detection complete! Check outputs in out/ directory:"
echo "  - out/measurements.csv  (per-specimen data)"
echo "  - out/summary.json      (full summary)"
echo "  - out/overlay.png       (visualization)"
echo "  - out/masks/            (individual masks)"
echo "================================================"
