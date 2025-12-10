# Collembola Analysis Pipeline - Complete Redesign

## Problem Statement

We need an **end-to-end automated pipeline** that:
1. Segments all objects in plate images using SAM
2. Classifies which segments are collembolas (organisms) vs junk/artifacts
3. Counts organisms per plate
4. Estimates morphological measurements and volume for each organism
5. Exports results to CSV for downstream analysis

## Current Assets

### Data we have:
- **Full plate images**: `data/slike/*.jpg` (10K×10K pixels, ~300MB each)
- **Labeled bounding boxes**: `data/collembolas_table.csv` (692 manual annotations)
- **Labeled crops**: `data/crops_dataset.csv` (positive + negative examples)
- **Template crops**: `data/organism_templates/` (214 organism images)
- **Physical ruler**: Printed on plates for pixel→µm calibration

### What went wrong with template matching:
- ❌ Too slow (214 templates × 5 scales = 1070 iterations, 10+ minutes)
- ❌ Poor recall (7 detections out of 692 = 1% recall)
- ❌ Wrong approach: Template matching assumes organisms look identical
- ❌ Reality: Organisms vary in size, orientation, pose, lighting

## New Approach: SAM AutoMask + Classifier

### Core Insight:
1. **SAM AutoMask Generator** segments EVERYTHING (organisms + junk)
2. **ResNet18 classifier** filters out junk, keeps only organisms
3. **Volume estimator** computes morphology from accepted masks

This is analogous to:
- Object detection (SAM) + Classification (ResNet) = Two-stage detector
- Similar to R-CNN family but using SAM instead of region proposals

---

## Implementation Plan

### Phase 1: Data Preparation & Calibration

**1.1 Reorganize data structure**
```
data/
  plates/              # Full plate images
  crops/               # All crop images
  annotations/
    collembolas_table.csv
    crops_dataset.csv
  calibration/
    ruler_measurements.json
```

**1.2 Pixel calibration**
- Measure ruler on representative plate
- Calculate `px_to_um` conversion factor
- Save to config file

**1.3 Build unified training dataset**
- Combine `collembolas_table.csv` (all positive) + `crops_dataset.csv` (pos+neg)
- Create stratified train/val split (80/20)
- Output: `train.csv`, `val.csv` with columns: `img_path, label`

### Phase 2: Train Organism Classifier

**2.1 Create PyTorch Dataset**
- Load images from `train.csv` / `val.csv`
- Transforms:
  - Train: Resize(128×128), RandomHFlip, RandomVFlip, RandomRotation(15°), ColorJitter, ToTensor, Normalize
  - Val: Resize(128×128), ToTensor, Normalize

**2.2 Train ResNet18**
- Start from ImageNet pretrained weights
- Replace final FC: 2 classes (organism vs junk)
- Loss: CrossEntropyLoss
- Optimizer: Adam(lr=1e-4)
- Train 10-20 epochs
- Save best model by val_acc → `models/classifier_resnet18.pt`

**2.3 Determine classification threshold**
- Evaluate on val set
- Plot precision-recall curve
- Choose threshold T that maximizes recall while keeping precision acceptable
- Target: 95%+ recall, 80%+ precision
- Save threshold to config

### Phase 3: SAM Segmentation Pipeline

**3.1 Configure SAM AutoMaskGenerator**
Parameters tuned for small thin objects:
```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b.pth")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # Grid density
    pred_iou_thresh=0.85,         # Quality threshold
    stability_score_thresh=0.9,   # Stability threshold
    min_mask_region_area=20,      # Min area (px²)
    box_nms_thresh=0.7,           # NMS for overlaps
)
```

**3.2 Process each plate**
For each image in `data/plates/`:
1. Load image
2. Run `mask_generator.generate(image)`
3. Returns list of masks, each with:
   - `segmentation`: binary mask (H×W)
   - `bbox`: [x, y, w, h]
   - `area`: pixel count
   - `predicted_iou`: SAM's confidence

### Phase 4: Classification & Filtering

**4.1 Extract crops from SAM masks**
For each mask:
1. Get bbox `[x, y, w, h]`
2. Add 5px padding (clip to image bounds)
3. Crop from original image
4. Convert to PIL Image

**4.2 Classify each crop**
```python
crop_tensor = val_transform(crop).unsqueeze(0)
with torch.no_grad():
    logits = classifier(crop_tensor)
    probs = F.softmax(logits, dim=1)
    p_organism = probs[0, 1].item()

if p_organism >= threshold:
    # Accept this mask as organism
    accepted_masks.append(mask)
```

### Phase 5: Morphology & Volume Estimation

**5.1 Extract measurements from accepted masks**
For each organism mask:
```python
from skimage.measure import regionprops

props = regionprops(mask['segmentation'].astype(int))[0]

# Extract properties
area_px = props.area
major_axis_px = props.major_axis_length  # Length
minor_axis_px = props.minor_axis_length  # Width
eccentricity = props.eccentricity
solidity = props.solidity
```

**5.2 Convert to physical units**
```python
length_um = major_axis_px * px_to_um
width_um = minor_axis_px * px_to_um
```

**5.3 Compute volume (prolate spheroid approximation)**
```python
# V = (4/3)π × (L/2) × (W/2)²
import numpy as np

a = length_um / 2  # Semi-major axis
b = width_um / 2   # Semi-minor axis
volume_um3 = (4/3) * np.pi * a * (b ** 2)
```

**5.4 Build per-organism record**
```python
organism_data = {
    'plate_name': plate_name,
    'organism_id': idx,
    'bbox_x': x,
    'bbox_y': y,
    'bbox_w': w,
    'bbox_h': h,
    'area_px': area_px,
    'major_axis_px': major_axis_px,
    'minor_axis_px': minor_axis_px,
    'length_um': length_um,
    'width_um': width_um,
    'volume_um3': volume_um3,
    'eccentricity': eccentricity,
    'solidity': solidity,
    'p_organism': p_organism,
}
```

### Phase 6: Output & Visualization

**6.1 Per-plate CSV export**
Save to `outputs/csv/<plate_name>_organisms.csv`:
```
plate_name,organism_id,bbox_x,bbox_y,bbox_w,bbox_h,area_px,length_um,width_um,volume_um3,p_organism
```

**6.2 Summary statistics**
```python
summary = {
    'plate_name': plate_name,
    'n_organisms': len(organisms),
    'total_volume_um3': sum(o['volume_um3'] for o in organisms),
    'mean_volume_um3': np.mean([o['volume_um3'] for o in organisms]),
    'median_volume_um3': np.median([o['volume_um3'] for o in organisms]),
    'mean_length_um': np.mean([o['length_um'] for o in organisms]),
    'mean_width_um': np.mean([o['width_um'] for o in organisms]),
}
```

**6.3 Debug overlay visualization**
```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(image)

for org in organisms:
    # Draw green box for accepted organisms
    rect = Rectangle(
        (org['bbox_x'], org['bbox_y']),
        org['bbox_w'], org['bbox_h'],
        linewidth=2, edgecolor='green', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add organism ID
    ax.text(
        org['bbox_x'], org['bbox_y'] - 5,
        f"{org['organism_id']}",
        color='green', fontsize=8, weight='bold'
    )

plt.savefig(f'outputs/overlays/{plate_name}_overlay.png', dpi=150, bbox_inches='tight')
```

### Phase 7: Validation & Quality Control

**7.1 Compare against ground truth**
For plates with manual annotations (`collembolas_table.csv`):
```python
# IoU matching between detected boxes and ground truth boxes
def compute_iou(box1, box2):
    # Standard IoU calculation
    pass

# Match detections to ground truth
matched, missed, false_positives = match_boxes(detections, ground_truth, iou_threshold=0.5)

metrics = {
    'precision': len(matched) / (len(matched) + len(false_positives)),
    'recall': len(matched) / (len(matched) + len(missed)),
    'f1': 2 * precision * recall / (precision + recall),
}
```

**7.2 Iteration loop**
1. Run pipeline on validation plates
2. Check metrics (target: recall > 90%, precision > 80%)
3. If recall too low:
   - Lower classifier threshold T
   - Retune SAM parameters (lower `pred_iou_thresh`)
   - Add more training data
4. If precision too low:
   - Raise classifier threshold T
   - Add hard negatives to training set
   - Retrain classifier
5. Repeat until satisfied

---

## Module Structure

```
collembola_pipeline/
  __init__.py
  config.py              # Configuration constants
  calibration.py         # Pixel calibration utilities
  data_prep.py           # Build training datasets
  train_classifier.py    # Train ResNet18
  segment.py             # SAM segmentation
  classify.py            # Apply classifier to crops
  morphology.py          # Volume & measurement calculation
  analyze_plate.py       # Main pipeline orchestrator
  visualize.py           # Create overlays
  validate.py            # Compare to ground truth
  
scripts/
  01_calibrate.py        # Interactive ruler calibration
  02_prepare_data.py     # Build train/val splits
  03_train.py            # Train classifier
  04_process_plates.py   # Run full pipeline
  05_validate.py         # Quality control
  
outputs/
  csv/                   # Per-plate organism CSVs
  overlays/              # Debug visualizations
  summary.csv            # All plates summary
  metrics.json           # Validation metrics
```

---

## Success Criteria

### Minimum Viable Pipeline (MVP):
- ✅ Detect 80%+ of organisms (recall ≥ 0.80)
- ✅ <20% false positives (precision ≥ 0.80)
- ✅ Volume estimates within ±20% of manual measurements (on subset)
- ✅ Process one plate in <5 minutes

### Stretch Goals:
- 🎯 Recall ≥ 0.95, Precision ≥ 0.90
- 🎯 Adaptive thresholding per plate (auto-tune based on image stats)
- 🎯 Multi-class classification (live vs dead organisms)
- 🎯 Batch processing with parallel GPU execution

---

## Next Steps

1. **Archive current template-matching approach**
   - Move `sam_templates.py` to `archive_old_scripts/`
   - Archive all shell scripts (`detect_*.sh`, `test_*.sh`)
   - Archive documentation (`CHANGELOG.md`, `DETECTION_METHODS.md`, etc.)

2. **Create new project structure**
   - Set up `collembola_pipeline/` module
   - Create `scripts/` for runnable entry points
   - Update `AGENTS.md` with new guidelines

3. **Start with Phase 1: Data Preparation**
   - Implement calibration utility
   - Build unified training dataset
   - Verify data quality

4. **Proceed sequentially through phases**
   - Each phase should be tested independently
   - Commit after each working phase
   - Document decisions and parameter choices

---

## Key Advantages of New Approach

1. **Scalable**: SAM finds all objects automatically, no manual template curation
2. **Robust**: Classifier learns from diverse examples, handles variation
3. **Fast**: SAM AutoMask is optimized, ResNet inference is milliseconds
4. **Validated**: Clear metrics against ground truth
5. **Maintainable**: Modular design, each component testable
6. **Extensible**: Easy to add new features (multi-class, tracking, etc.)

---

## Timeline Estimate

- Phase 1 (Data prep): 2-4 hours
- Phase 2 (Train classifier): 4-6 hours (including experimentation)
- Phase 3 (SAM setup): 2-3 hours
- Phase 4 (Integration): 3-4 hours
- Phase 5 (Morphology): 2-3 hours
- Phase 6 (Output): 2-3 hours
- Phase 7 (Validation): 3-5 hours

**Total: ~20-30 hours** for full working pipeline

---

**Author**: OpenCode  
**Date**: 2025-11-29  
**Status**: Planning - Ready for Implementation
