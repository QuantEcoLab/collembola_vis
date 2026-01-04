# Collembola Detection Model - Tiled Inference for Ultra-High-Resolution Microscope/Magnifier Images

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11](https://img.shields.io/badge/YOLO-11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

High-performance YOLO-based detection pipeline for collembola organisms in ultra-high-resolution microscope images (10K×10K pixels) using tiled inference and multi-GPU training.

## 🎯 Overview

This pipeline uses a **tiled YOLO approach** to detect collembola organisms in large microscope images:
- **Input**: Ultra-high-resolution images (10408×10338 pixels)
- **Method**: Tile images into 1280×1280 patches with overlap, run YOLO detection, merge with NMS
- **Performance**: 99.2% mAP@0.5, 97.8% precision, 97.1% recall
- **Training**: Multi-GPU support (4× Quadro RTX 8000)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n collembola python=3.11
conda activate collembola

# Install dependencies
pip install ultralytics torch pandas pillow numpy read-roi
```

### 2. Run Detection on Full Image

```bash
python scripts/infer_tiled.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --model models/yolo11n_tiled_best.pt \
    --conf 0.25 \
    --device 0
```

**Output** (in `infer_tiled_output/`):
- `K1_Fe2O3001 (1)_detections.csv` - All detections with coordinates and confidence
- `K1_Fe2O3001 (1)_overlay.jpg` - Visualization with bounding boxes
- `K1_Fe2O3001 (1)_metadata.json` - Inference parameters and statistics

### 3. Measure Morphological Properties

**Option A: Fast Method (Recommended)** - 186× faster, ellipse fitting
```bash
python scripts/measure_organisms_fast.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --detections infer_tiled_output/K1_detections.csv \
    --um-per-pixel 8.57
```
- **Speed**: ~4 seconds for 746 organisms (178 org/sec)
- **Method**: Adaptive threshold → ellipse fitting → morphology
- **Use case**: Production, large-scale processing

**Option B: SAM Method** - Maximum accuracy, slower
```bash
python scripts/measure_organisms.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --detections infer_tiled_output/K1_detections.csv \
    --um-per-pixel 8.57 \
    --device cuda
```
- **Speed**: ~13 minutes for 746 organisms (1 org/sec)
- **Method**: SAM segmentation → precise masks → morphology
- **Use case**: Research, maximum precision needed

**Output** (in `measurements/`):
- `K1_measurements.csv` - Body length, width, area, volume for each organism
- `K1_measurements_metadata.json` - Summary statistics (mean length, total volume, etc.)

**Measurements include:**
- Body length (µm) - major axis from segmentation
- Body width (µm) - minor axis from segmentation
- Area (µm²) - segmented area in square micrometers
- Volume (µm³) - cylinder model: V = π × r² × h
- Morphological features: eccentricity, solidity, perimeter

### 4. Batch Process Multiple Plates

```bash
python scripts/process_plate_batch.py \
    --images "data/slike/*.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --um-per-pixel 8.57 \
    --output-dir outputs/batch_20251210
```

**Output structure:**
```
outputs/batch_20251210/
├── detections/           # YOLO detection CSVs
├── measurements/         # Morphological measurements CSVs
├── overlays/             # Visualization images
└── batch_config.json     # Batch processing configuration
```

## 📊 Performance

### Model Metrics (Best Epoch: 82)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **99.2%** |
| **mAP@0.5:0.95** | **85.2%** |
| **Precision** | **97.8%** |
| **Recall** | **97.1%** |

### Training Dataset

- **Source**: 14,125 ImageJ ROI annotations from 20 microscope plates
- **Tiled Dataset**: 
  - Training: 1,246 tiles (16,701 annotations)
  - Validation: 200 tiles (3,950 annotations)
- **Tile Size**: 1280×1280 with 256px overlap

### Comparison: Tiled vs Downscaled Approach

| Approach | mAP@0.5 | Precision | Recall |
|----------|---------|-----------|--------|
| **Downscaled** (10K→1280) | 39.6% | 56.4% | 23.7% |
| **Tiled** (1280 tiles) | **99.2%** | **97.8%** | **97.1%** |

**Result**: Tiled approach is **2.5× better** while preserving all image details!

## 🔧 Pipeline Components

### Core Scripts

**Detection Pipeline:**
1. **`scripts/convert_imagej_rois.py`** - Extract ImageJ ROI annotations to CSV
2. **`scripts/create_tiled_dataset.py`** - Tile images and create YOLO dataset
3. **`scripts/train_yolo_tiled.py`** - Multi-GPU training script
4. **`scripts/infer_tiled.py`** - Tiled inference with NMS merging

**Measurement Pipeline:**
5. **`scripts/calibrate_ruler.py`** - Interactive ruler calibration (µm/pixel)
6. **`scripts/measure_organisms_fast.py`** - Fast ellipse fitting (178 org/sec) ⚡ **Recommended**
7. **`scripts/measure_organisms.py`** - SAM segmentation (1 org/sec, max accuracy)
8. **`scripts/process_plate_batch.py`** - Batch process multiple plates (detection + measurement)

### Key Features

**Detection:**
- ✅ **Tiled Processing**: Handles ultra-high-resolution images without downscaling
- ✅ **Multi-GPU Training**: Distributed training on 4 GPUs with DDP
- ✅ **Overlap Handling**: 256px overlap between tiles prevents edge artifacts
- ✅ **Global NMS**: Merges detections across tile boundaries
- ✅ **Confidence Filtering**: Adjustable confidence thresholds
- ✅ **Metadata Tracking**: Full provenance of tiles and detections

**Measurement:**
- ✅ **SAM Segmentation**: Precise organism masks using Segment Anything Model
- ✅ **Morphological Analysis**: Length, width, area, volume, eccentricity, solidity
- ✅ **Cylinder Volume Model**: V = π × r² × h for accurate volume estimation
- ✅ **Auto-Calibration**: Interactive ruler detection for µm/pixel calibration
- ✅ **Batch Processing**: Process multiple plates with single command

## 📁 Project Structure

```
collembola_vis/
├── scripts/                          # Core pipeline scripts
│   ├── convert_imagej_rois.py       # ROI extraction from ImageJ
│   ├── create_tiled_dataset.py      # Tiled dataset creation
│   ├── train_yolo_tiled.py          # Multi-GPU training
│   ├── infer_tiled.py               # Tiled inference
│   ├── calibrate_ruler.py           # Interactive ruler calibration
│   ├── measure_organisms.py         # SAM segmentation + measurements
│   ├── measure_organisms_fast.py    # Fast ellipse-based measurements
│   ├── process_plate_batch.py       # Batch processing
│   └── monitor_batch.sh             # Batch monitoring utility
├── collembola_pipeline/             # Python package for pipeline modules
│   ├── detect_organisms.py          # Detection utilities
│   ├── detect_plate.py              # Plate detection
│   ├── morphology.py                # Morphological analysis
│   └── ...                          # Other pipeline modules
├── models/
│   └── yolo11n_tiled_best.pt        # Best trained model (99.2% mAP@0.5)
├── docs/                            # Documentation
│   ├── QUICKSTART.md                # Quick start guide
│   ├── WORKFLOW.md                  # Detailed workflow
│   ├── MEASUREMENT_METHODS.md       # Measurement documentation
│   ├── PERFORMANCE.md               # Performance metrics
│   ├── TROUBLESHOOTING.md           # Common issues
│   ├── MODEL_CARD.md                # Model card
│   ├── HUGGINGFACE_README.md        # HuggingFace documentation
│   └── CHANGELOG.md                 # Version history
├── data/                            # Data directory (see .gitignore)
│   ├── training_data/               # ImageJ ROI annotations
│   ├── annotations/                 # Extracted ROI CSV
│   ├── yolo_tiled/                  # Tiled YOLO dataset
│   └── slike/                       # Production images
├── outputs/                         # Generated outputs (see .gitignore)
├── runs/                            # Training runs (see .gitignore)
├── checkpoints/                     # Model checkpoints (see .gitignore)
└── archive/                         # Historical development materials
    ├── scripts/                     # Deprecated scripts
    ├── models/                      # Old model checkpoints
    ├── training_runs/               # Previous training runs
    ├── outputs/                     # Old inference outputs
    ├── datasets/                    # Old datasets
    ├── template_approach/           # SAM template experiments
    ├── unused/                      # Miscellaneous archived files
    └── README.md                    # Archive documentation
```

## 📏 Measurement Workflow

### Complete Pipeline: Detection → Measurement

```bash
# Step 1: Run tiled YOLO detection
python scripts/infer_tiled.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --model models/yolo11n_tiled_best.pt \
    --output infer_tiled_output \
    --device cuda

# Step 2: Calibrate ruler (one-time per microscope setup)
python scripts/calibrate_ruler.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --ruler-mm 10

# Step 3: Measure organisms (fast method - recommended)
python scripts/measure_organisms_fast.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --detections infer_tiled_output/K1_Fe2O3001_(1)_detections.csv \
    --um-per-pixel 8.57

# Alternative: SAM method for maximum accuracy (186× slower)
# python scripts/measure_organisms.py \
#     --image "data/slike/K1_Fe2O3001 (1).jpg" \
#     --detections infer_tiled_output/K1_Fe2O3001_(1)_detections.csv \
#     --um-per-pixel 8.57 \
#     --device cuda
```

### Batch Processing

Process all plates in a directory:

```bash
python scripts/process_plate_batch.py \
    --images "data/slike/*.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --um-per-pixel 8.57 \
    --output-dir outputs/batch_experiment_1 \
    --device cuda
```

### Measurement Output

**CSV Format** (`measurements/*.csv`):

| Column | Description | Unit |
|--------|-------------|------|
| `detection_id` | Unique organism ID | - |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | Bounding box coordinates | pixels |
| `centroid_x_px, centroid_y_px` | Organism centroid | pixels |
| `length_um` | Body length (major axis) | µm |
| `width_um` | Body width (minor axis) | µm |
| `area_um2` | Segmented area | µm² |
| `volume_um3` | Cylinder model volume | µm³ |
| `eccentricity` | Shape elongation (0-1) | - |
| `solidity` | Convexity measure | - |
| `confidence` | YOLO detection confidence | - |
| `mask_available` | SAM segmentation success | boolean |

**Example measurements:**
```csv
detection_id,length_um,width_um,area_um2,volume_um3
0,1960.89,673.99,1006929.58,699597349.21
1,2950.53,952.20,1774722.56,2101086027.68
2,2598.03,553.45,1071634.54,625021256.24
```

**Summary Statistics** (`measurements/*_metadata.json`):

```json
{
  "image_path": "data/slike/K1_Fe2O3001 (1).jpg",
  "num_organisms": 800,
  "mean_length_um": 2221.2,
  "mean_width_um": 719.6,
  "total_volume_um3": 9683963333.1,
  "um_per_pixel": 8.57
}
```

## 🎓 Training Your Own Model

### 1. Prepare Dataset

```bash
# Extract ImageJ ROIs to CSV
python scripts/convert_imagej_rois.py

# Create tiled YOLO dataset
python scripts/create_tiled_dataset.py
```

### 2. Train with Multi-GPU

```bash
# Train on all 4 GPUs
python scripts/train_yolo_tiled.py \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch 32 \
    --patience 30

# Train on single GPU
python scripts/train_yolo_tiled.py \
    --device 0 \
    --epochs 100 \
    --batch 16
```

**Training Configuration**:
- Model: YOLO11n (2.59M parameters)
- Image size: 1280×1280
- Batch size: 32 (8 per GPU on 4 GPUs)
- Optimizer: AdamW (auto-selected)
- Data augmentation: Mosaic, flip, HSV, scale
- Early stopping: Patience 30 epochs

### 3. Monitor Training

Results saved in `runs/detect/train_tiled_*/`:
- `weights/best.pt` - Best model checkpoint
- `results.csv` - Per-epoch metrics
- `confusion_matrix.png` - Validation confusion matrix
- `PR_curve.png` - Precision-Recall curve

## 🔬 Inference Options

### Basic Inference

```bash
python scripts/infer_tiled.py \
    --image path/to/image.jpg \
    --model models/yolo11n_tiled_best.pt
```

### Advanced Options

```bash
python scripts/infer_tiled.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --model models/yolo11n_tiled_best.pt \
    --tile-size 1280 \         # Tile size (must match training)
    --overlap 256 \            # Overlap between tiles
    --conf 0.25 \              # Confidence threshold
    --iou 0.5 \                # NMS IoU threshold
    --output results/ \        # Output directory
    --device 0                 # GPU device
```

### Output Format

**CSV** (`*_detections.csv`):
```csv
x1,y1,x2,y2,width,height,confidence,class
7440.3,2143.5,7612.7,2295.6,172.4,152.1,0.961,0
5898.0,3250.9,6080.5,3517.6,182.5,266.7,0.959,0
```

**Metadata** (`*_metadata.json`):
```json
{
  "image_size": [10408, 10338],
  "num_tiles": 100,
  "detections_before_nms": 1214,
  "detections_after_nms": 800,
  "conf_threshold": 0.25
}
```

## 🛠️ Technical Details

### Tiling Strategy

- **Tile Size**: 1280×1280 pixels
- **Overlap**: 256 pixels (20%)
- **Stride**: 1024 pixels (tile_size - overlap)
- **Edge Handling**: Tiles adjusted at image boundaries

### Global NMS Algorithm

1. Collect all detections from all tiles
2. Convert tile coordinates to full-image coordinates
3. Sort detections by confidence (descending)
4. Apply NMS with IoU threshold (default: 0.5)
5. Keep highest-confidence non-overlapping detections

### Multi-GPU Training

- **Framework**: PyTorch DistributedDataParallel (DDP)
- **Batch Distribution**: Even split across GPUs
- **Synchronization**: Gradient averaging across devices
- **Memory**: ~7.2GB per GPU during training

## 📈 Example Results

**K1_Fe2O3001 (1).jpg** (10408×10338 pixels):
- Tiles processed: 100
- Raw detections: 1,214
- Final detections after NMS: 800
- Processing time: ~2-3 minutes on single GPU

## 🔍 Troubleshooting

### Low Detection Count

**Solution**: Lower confidence threshold
```bash
python scripts/infer_tiled.py --conf 0.15  # Instead of default 0.25
```

### Out of Memory During Inference

**Solution**: Use smaller batch or single-tile processing
```bash
python scripts/infer_tiled.py --device 0  # Use single GPU
```

### Training Crashes with Multi-GPU

**Solution**: Ensure batch size is multiple of GPU count
```bash
# For 4 GPUs, use batch=32, 16, 8, etc.
python scripts/train_yolo_tiled.py --batch 16 --device 0,1,2,3
```

## 📚 Data Sources

### Training Data
- **ImageJ ROI Annotations**: `data/training_data/Collembola_ROI setovi/`
- **Annotation Type**: "bez antene i furce" (without antennae and furca)
- **Total Annotations**: 14,125 ROIs from 20 plates (15 Fe2O3 + 5 Mikroplastika)

### Production Data
- **Images**: `data/slike/`
- **Reference Annotations**: `data/collembolas_table.csv` (manual counts from 3 plates)

## 🎯 Roadmap & Future Improvements

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### High Priority
- [ ] Implement instance segmentation (YOLO-seg) for precise boundaries
- [ ] Web interface for batch processing
- [ ] Automated quality control and validation
- [ ] Export to multiple formats (COCO, Pascal VOC, ImageJ ROIs)

### Enhancements
- [ ] Multi-species classification
- [ ] Interactive annotation tools
- [ ] Cloud deployment support (Docker, Kubernetes)
- [ ] Real-time video processing
- [ ] Enhanced visualization dashboard

### Documentation
- [ ] Tutorial notebooks
- [ ] Video walkthroughs
- [ ] API documentation
- [ ] Use case examples

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to report issues
- How to submit pull requests
- Development setup
- Code style guidelines

## 📖 Citation

**Project**: Collembola Detection Pipeline  
**Repository**: https://github.com/QuantEcoLab/collembolae_vis  
**Authors**: Jana Zovko, Domagoj K. Hackenberger  
**Method**: Tiled YOLO11n with Multi-GPU Training

## 📝 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.

**Why AGPL-3.0?** This project uses Ultralytics YOLO, which is licensed under AGPL-3.0. As a derivative work, this project must also use AGPL-3.0.

**Key Points:**
- You can freely use, modify, and distribute this software
- If you modify and distribute this software, you must release your modifications under AGPL-3.0
- If you run a modified version on a server, you must make the source code available to users
- For commercial use without AGPL requirements, contact Ultralytics for commercial licensing options

---

## 🗂️ Archived Components

Previous approaches (SAM-based, classical CV, downscaled YOLO) and development artifacts are archived in the `archive/` directory. This includes:

- **scripts/** - Deprecated scripts (SAM templates, classical segmentation methods)
- **training_runs/** - Non-tiled training attempts
- **models/** - Downscaled YOLO models (39.6% mAP)
- **outputs/** - Previous inference results
- **template_approach/** - SAM template-based detection experiments
- **zenodo_upload/** - Model package for Zenodo repository
- **upload_scripts/** - Scripts for uploading to HuggingFace and Zenodo

See `archive/README.md` for detailed information about archived materials. These are kept for historical reference but **not recommended for production use**.
