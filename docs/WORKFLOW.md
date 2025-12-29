# Collembola Measurement Workflow

Complete pipeline for processing collembola plate images with automatic ruler calibration, YOLO detection, fast morphological measurements, and validation visualizations.

## Quick Start

Process a single image with one command:

```bash
conda activate collembola
python scripts/process_single_image.py "data/slike/K1_Fe2O3001 (1).jpg" --output output
```

This runs the complete pipeline:
1. **Automatic ruler calibration** (8.666 µm/pixel)
2. **YOLO detection** (746 organisms, conf=0.6)
3. **Fast measurements** (178 organisms/sec)
4. **Validation visualizations** (overview + samples)

**Processing time**: ~30-60 seconds per 10K×10K image

---

## Output Files (Flat Structure)

All files are saved in the output directory with the image name as prefix:

```
output/
├── K1_Fe2O3001 (1)_calibration.json          # Ruler calibration data
├── K1_Fe2O3001 (1)_ruler_analysis.png        # Visual validation of ruler detection
├── K1_Fe2O3001 (1)_detections.csv            # YOLO bounding boxes
├── K1_Fe2O3001 (1)_measurements.csv          # Morphological measurements (main output)
├── K1_Fe2O3001 (1)_metadata.json             # Processing metadata
├── K1_Fe2O3001 (1)_overview.png              # All organisms with bounding boxes
├── K1_Fe2O3001 (1)_samples.png               # Grid of 50 sample organisms
├── K1_Fe2O3001 (1)_overlay.jpg               # YOLO detection overlay
└── K1_Fe2O3001 (1)_measurements_metadata.json # Measurement statistics
```

### Key Output: `*_measurements.csv`

Contains one row per organism with:

| Column | Description | Units |
|--------|-------------|-------|
| `detection_id` | Sequential organism ID | - |
| `length_um` | Major axis length | µm |
| `width_um` | Minor axis width | µm |
| `area_um2` | Ellipse area | µm² |
| `volume_um3` | Cylinder volume (V = πr²h) | µm³ |
| `confidence` | YOLO detection confidence | 0-1 |
| `eccentricity` | Shape eccentricity | 0-1 |
| `solidity` | Shape solidity | 0-1 |
| `centroid_x_px`, `centroid_y_px` | Centroid position | pixels |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | Bounding box | pixels |

---

## Pipeline Steps

### Step 1: Automatic Ruler Calibration

**Script**: `scripts/calibrate_ruler_auto.py`

Detects the 10cm ruler in the image and calculates pixels-per-micrometer calibration.

**Ruler specifications**:
- Location: Fixed region (x=4000-9000, y=1500-3000)
- Tick marks: **WHITE/BRIGHT** (not dark)
- Major ticks: 5mm (0.5cm) intervals
- Minor ticks: ~1mm intervals (for validation)

**Output**: `*_calibration.json`, `*_ruler_analysis.png`

**Typical result**: **8.666 µm/pixel** (577 px/cm)

**Fallback**: If calibration fails, uses default 8.666 µm/pixel

```bash
# Run standalone
python scripts/calibrate_ruler_auto.py --image "data/slike/image.jpg" --output output/
```

---

### Step 2: YOLO Detection

**Script**: `scripts/infer_tiled.py`

Detects collembola organisms using YOLO11n model with tiled inference.

**Model**: `models/yolo11n_tiled_best.pt` (99.2% mAP@0.5)

**Method**: 
- Tile size: 1280×1280 pixels
- Overlap: 256 pixels
- Global NMS (IoU=0.5) to remove duplicates

**Output**: `*_detections.csv`, `*_overlay.jpg`, `*_metadata.json`

**Confidence threshold**: 0.6 (adjustable with `--conf`)

```bash
# Run standalone
python scripts/infer_tiled.py \
  --image "data/slike/image.jpg" \
  --output output/ \
  --conf 0.6
```

---

### Step 3: Fast Measurements

**Script**: `scripts/measure_organisms_fast.py`

Measures organism morphology using ellipse fitting (orientation-invariant via eigenvalue decomposition).

**Method**:
1. Extract ROI from each bounding box
2. Convert to grayscale and threshold
3. Find largest connected component
4. Fit ellipse using covariance matrix eigenvalues
5. Calculate length, width, area, volume

**Performance**: **178 organisms/second**

**Volume calculation**: Cylinder model (V = π × r² × h)
- r = minor axis radius (width/2)
- h = major axis length

**Output**: `*_measurements.csv`, `*_measurements_metadata.json`

```bash
# Run standalone
python scripts/measure_organisms_fast.py \
  --image "data/slike/image.jpg" \
  --detections output/image_detections.csv \
  --um-per-pixel 8.666 \
  --output output/image_measurements.csv
```

---

### Step 4: Validation Visualizations

**Script**: `scripts/visualize_measurements.py`

Creates validation visualizations with measurement overlays.

**Outputs**:

1. **Overview** (`*_overview.png`):
   - All organisms with bounding boxes
   - Color-coded by size:
     - 🟢 Green: small (<500 µm)
     - 🟡 Yellow: medium (500-1500 µm)
     - 🔴 Red: large (>1500 µm)
   - Labels: length×width (µm)

2. **Samples** (`*_samples.png`):
   - Grid of 50 organisms (stratified by size)
   - Each crop shows:
     - Detection ID
     - Length × width (µm)
     - Area (µm²)
     - Volume (µm³)
     - Confidence score

```bash
# Run standalone
python scripts/visualize_measurements.py \
  --image "data/slike/image.jpg" \
  --detections output/image_detections.csv \
  --measurements output/image_measurements.csv \
  --output output/
```

---

## Command-Line Options

### Master Pipeline Script

```bash
python scripts/process_single_image.py IMAGE [OPTIONS]
```

**Arguments**:
- `IMAGE`: Path to input image (required)

**Options**:
- `--output DIR`: Output directory (default: `output/`)
- `--conf FLOAT`: YOLO confidence threshold (default: `0.6`)
- `--default-cal FLOAT`: Fallback calibration if ruler detection fails (default: `8.666` µm/px)

**Examples**:

```bash
# Basic usage
python scripts/process_single_image.py "data/slike/K1.jpg"

# Custom output directory
python scripts/process_single_image.py "data/slike/K1.jpg" --output results/

# Lower confidence threshold (more detections)
python scripts/process_single_image.py "data/slike/K1.jpg" --conf 0.5

# Custom fallback calibration
python scripts/process_single_image.py "data/slike/K1.jpg" --default-cal 9.0
```

---

## Typical Results

**Test image**: `K1_Fe2O3001 (1).jpg` (10408×10338 pixels)

```
============================================================
PROCESSING COMPLETE
============================================================

✓ Calibration: 8.666 µm/px

✓ Organisms detected: 746
  Mean length:     888.7 µm
  Mean width:      235.0 µm
  Mean area:       83670.6 µm²
  Mean volume:     128228552.2 µm³
  Mean confidence: 0.859

✓ Total processing time: 38.4 seconds (0.6 minutes)
============================================================
```

---

## Batch Processing

To process multiple images, use a simple loop:

```bash
#!/bin/bash
conda activate collembola

for image in data/slike/*.jpg; do
  echo "Processing: $image"
  python scripts/process_single_image.py "$image" --output output/
done
```

Or create a Python script:

```python
from pathlib import Path
import subprocess

image_dir = Path("data/slike")
output_dir = Path("output")

for image_path in image_dir.glob("*.jpg"):
    print(f"Processing: {image_path.name}")
    subprocess.run([
        "python", "scripts/process_single_image.py",
        str(image_path),
        "--output", str(output_dir)
    ])
```

---

## Environment Setup

### First-Time Setup

```bash
# Create conda environment
conda create -n collembola python=3.11
conda activate collembola

# Install dependencies
pip install pandas scikit-image matplotlib tqdm numpy openpyxl pillow opencv-python scipy ultralytics torch
```

### Daily Usage

```bash
# Activate environment
conda activate collembola

# Process images
python scripts/process_single_image.py "data/slike/image.jpg"
```

---

## Troubleshooting

### Calibration Fails

**Symptoms**: "WARNING: Calibration failed, will use default 8.666 µm/px"

**Causes**:
1. Ruler not in expected location (x=4000-9000, y=1500-3000)
2. Ruler ticks too faint/damaged
3. Incorrect ruler type (not 10cm with 5mm major ticks)

**Solutions**:
1. Check `*_ruler_analysis.png` to see what was detected
2. Use manual calibration as fallback:
   ```bash
   python scripts/calibrate_ruler.py --image "image.jpg" --interactive
   ```
3. Specify custom fallback calibration if you know the scale:
   ```bash
   python scripts/process_single_image.py "image.jpg" --default-cal 9.0
   ```

### Detection Issues

**Too few detections**:
- Lower confidence threshold: `--conf 0.5`

**Too many false positives**:
- Raise confidence threshold: `--conf 0.7`

**Missing organisms**:
- Check `*_overlay.jpg` to visually inspect detections
- Model may need retraining on edge cases

### Measurement Issues

**Incorrect sizes**:
- Verify calibration in `*_calibration.json`
- Check ruler detection in `*_ruler_analysis.png`
- If calibration failed, measurements will be off

**Memory errors**:
- Large images (>10K×10K) may require more RAM
- Close other applications or use a machine with more memory

---

## File Organization

```
collembola_vis/
├── data/
│   └── slike/                  # Input images
├── output/                     # Output directory (created automatically)
├── models/
│   └── yolo11n_tiled_best.pt  # YOLO model weights
├── scripts/
│   ├── process_single_image.py        # ⭐ Master pipeline
│   ├── calibrate_ruler_auto.py        # Step 1: Calibration
│   ├── infer_tiled.py                 # Step 2: Detection
│   ├── measure_organisms_fast.py      # Step 3: Measurements
│   └── visualize_measurements.py      # Step 4: Visualization
└── WORKFLOW.md                 # This file
```

---

## Performance Notes

- **Detection**: ~2-3 minutes (YOLO tiled inference on 10K×10K image)
- **Measurement**: ~4 seconds (746 organisms at 178 org/sec)
- **Visualization**: ~20-30 seconds (rendering overlays)
- **Total**: ~30-60 seconds per plate

**GPU acceleration**: Automatically used if available (speeds up detection 5-10×)

---

## Citation & Model Info

**YOLO Model**:
- Architecture: YOLO11n (nano)
- Training: Tiled dataset (1280×1280)
- Performance: 99.2% mAP@0.5
- Weights: `models/yolo11n_tiled_best.pt`

**Measurement Method**:
- Ellipse fitting via eigenvalue decomposition
- Orientation-invariant (handles organisms at any angle)
- Volume: Cylinder approximation

---

## Next Steps

After processing, you can:

1. **Analyze measurements**: Load `*_measurements.csv` in Excel/Pandas
2. **Quality control**: Review `*_overview.png` and `*_samples.png`
3. **Statistical analysis**: Calculate size distributions, treatment effects, etc.
4. **Export**: Measurements are in standard CSV format for downstream analysis

For questions or issues, see `TROUBLESHOOTING.md` or check `CHANGELOG.md` for recent updates.
