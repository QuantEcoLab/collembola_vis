#!/bin/bash
# Detect collembolas automatically with optimized template matching
# Then validate against 692 ground truth annotations

set -e

echo ""
echo "============================================================"
echo "🔬 Automated Collembola Detection with Validation"
echo "============================================================"
echo ""

source ~/miniforge3/etc/profile.d/conda.sh
conda activate collembola

mkdir -p out/auto_detect

echo "Running automatic detection with optimized parameters..."
echo ""

# Optimized parameters for HIGH RECALL:
# - Lower NCC threshold (0.5 instead of 0.65)
# - Multiple scales (0.7, 0.85, 1.0, 1.15, 1.3)
# - More prompts (1000 instead of 100)
# - Less aggressive filtering
# - No template subsampling (use all 214 templates)

python sam_templates.py \
    "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --downscale-max-side 2048 \
    --scale-factors 0.7,0.85,1.0,1.15,1.3 \
    --ncc-threshold 0.5 \
    --peak-distance 20 \
    --max-prompts 1000 \
    --min-box-overlap 0.3 \
    --output out/auto_detect/measurements.csv \
    --json out/auto_detect/summary.json \
    --save-overlay out/auto_detect/overlay.png \
    --save-masks-dir out/auto_detect/masks \
    --auto-download \
    --allow-large-image

echo ""
echo "============================================================"
echo "✅ Automatic detection complete!"
echo ""
echo "Output files:"
echo "  - out/auto_detect/overlay.png"
echo "  - out/auto_detect/measurements.csv"
echo "  - out/auto_detect/summary.json"
echo ""
echo "Now validating against ground truth (692 annotations)..."
echo "============================================================"
echo ""

# Count detections
DETECTED=$(tail -n +2 out/auto_detect/measurements.csv | wc -l)
echo "📊 Detected: $DETECTED collembolas"
echo "📋 Ground truth: 692 collembolas"
echo "📈 Recall: $(python3 -c "print(f'{$DETECTED/692*100:.1f}%')")"
echo ""
echo "View overlay: eog out/auto_detect/overlay.png"
echo ""
