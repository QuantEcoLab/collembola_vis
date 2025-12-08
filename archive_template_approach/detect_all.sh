#!/bin/bash
# Fast detection with 50 templates but aggressive parameters
# Balances speed vs recall

set -e

echo ""
echo "============================================================"
echo "🔬 Fast Collembola Detection (50 templates)"
echo "============================================================"
echo ""

source ~/miniforge3/etc/profile.d/conda.sh
conda activate collembola

mkdir -p out/fast_detect

echo "Running fast detection with aggressive parameters..."
echo ""

# FAST but AGGRESSIVE parameters:
# - 50 templates (instead of 214)
# - Lower NCC threshold (0.45 instead of 0.65)
# - 7 scales for better size coverage
# - More prompts (2000 instead of 100)
# - Very loose filtering to maximize recall

python sam_templates.py \
    "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --downscale-max-side 2048 \
    --max-templates 50 \
    --scale-factors 0.6,0.7,0.85,1.0,1.15,1.3,1.5 \
    --ncc-threshold 0.45 \
    --peak-distance 15 \
    --max-prompts 2000 \
    --min-box-overlap 0.25 \
    --output out/fast_detect/measurements.csv \
    --json out/fast_detect/summary.json \
    --save-overlay out/fast_detect/overlay.png \
    --save-masks-dir out/fast_detect/masks \
    --auto-download \
    --allow-large-image

echo ""
echo "============================================================"
echo "✅ Fast detection complete!"
echo ""
echo "Output files:"
echo "  - out/fast_detect/overlay.png"
echo "  - out/fast_detect/measurements.csv"
echo "  - out/fast_detect/summary.json"
echo ""

# Count detections
DETECTED=$(tail -n +2 out/fast_detect/measurements.csv 2>/dev/null | wc -l)
echo "📊 Detected: $DETECTED collembolas"
echo "📋 Ground truth: 692 collembolas"
if [ $DETECTED -gt 0 ]; then
    echo "📈 Recall: $(python3 -c "print(f'{$DETECTED/692*100:.1f}%')")"
else
    echo "📈 Recall: 0.0%"
fi
echo ""
echo "View overlay: eog out/fast_detect/overlay.png"
echo ""
