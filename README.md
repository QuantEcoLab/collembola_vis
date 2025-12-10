# Collembola Detection Pipeline

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

1. **`scripts/convert_imagej_rois.py`** - Extract ImageJ ROI annotations to CSV
2. **`scripts/create_tiled_dataset.py`** - Tile images and create YOLO dataset
3. **`scripts/train_yolo_tiled.py`** - Multi-GPU training script
4. **`scripts/infer_tiled.py`** - Tiled inference with NMS merging

### Key Features

- ✅ **Tiled Processing**: Handles ultra-high-resolution images without downscaling
- ✅ **Multi-GPU Training**: Distributed training on 4 GPUs with DDP
- ✅ **Overlap Handling**: 256px overlap between tiles prevents edge artifacts
- ✅ **Global NMS**: Merges detections across tile boundaries
- ✅ **Confidence Filtering**: Adjustable confidence thresholds
- ✅ **Metadata Tracking**: Full provenance of tiles and detections

## 📁 Project Structure

```
collembola_vis/
├── scripts/                          # Active pipeline scripts
│   ├── convert_imagej_rois.py       # ROI extraction from ImageJ
│   ├── create_tiled_dataset.py      # Tiled dataset creation
│   ├── train_yolo_tiled.py          # Multi-GPU training
│   └── infer_tiled.py               # Tiled inference
├── models/
│   └── yolo11n_tiled_best.pt        # Best trained model (99.2% mAP@0.5)
├── data/
│   ├── training_data/               # ImageJ ROI annotations (20 plates)
│   ├── annotations/                 # Extracted ROI CSV
│   ├── yolo_tiled/                  # Tiled YOLO dataset
│   └── slike/                       # Production images for inference
├── runs/detect/
│   └── train_tiled_1280_20251210_115016/  # Training run with best model
├── archive_old_scripts/             # Deprecated scripts (SAM, classical methods)
├── archive_training_runs/           # Old training attempts
├── archive_models/                  # Old non-tiled models
└── archive_outputs/                 # Previous inference outputs
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

## 🎯 Future Improvements

- [ ] Implement instance segmentation (YOLO-seg) for precise boundaries
- [ ] Add morphological measurements (length, width, area)
- [ ] Export to multiple formats (COCO, Pascal VOC)
- [ ] Web interface for batch processing
- [ ] Automated quality control and validation

## 📖 Citation

**Project**: Collembola Detection Pipeline  
**Repository**: https://github.com/QuantEcoLab/collembolae_vis  
**Authors**: Jana Zovko, Domagoj K. Hackenberger  
**Method**: Tiled YOLO11n with Multi-GPU Training

## 📝 License

Research use only. See repository for details.

---

## 🗂️ Archived Components

Previous approaches (SAM-based, classical CV, downscaled YOLO) are archived in:
- `archive_old_scripts/` - SAM templates, classical segmentation methods
- `archive_training_runs/` - Non-tiled training attempts
- `archive_models/` - Downscaled YOLO models (39.6% mAP)
- `archive_outputs/` - Previous inference results

These are kept for reference but **not recommended for production use**.
