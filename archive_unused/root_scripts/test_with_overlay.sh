#!/bin/bash
# Quick test run with overlay generation
# This creates the overlay visualization you're missing

set -e

echo "🧪 Running quick test with overlay generation..."
echo ""

source ~/miniforge3/etc/profile.d/conda.sh
conda activate collembola

# Create fresh output directory
mkdir -p out/test
mkdir -p out/test/masks

# Run with minimal settings but WITH overlay
python sam_templates.py \
    "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --downscale-max-side 2048 \
    --scale-factors 1.0 \
    --ncc-threshold 0.65 \
    --max-prompts 50 \
    --output out/test/measurements.csv \
    --json out/test/summary.json \
    --save-overlay out/test/overlay.png \
    --save-masks-dir out/test/masks \
    --allow-large-image

echo ""
echo "✅ Test complete! Check your overlay:"
echo "   out/test/overlay.png"
echo ""
echo "Open with:"
echo "   xdg-open out/test/overlay.png"
echo "   # or"
echo "   eog out/test/overlay.png"
echo ""
