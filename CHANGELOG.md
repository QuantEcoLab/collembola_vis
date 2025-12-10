# Changelog

All notable changes to the Collembola Detection Pipeline.

## [2.0.0] - 2024-12-10 - Tiled YOLO Pipeline (MAJOR MILESTONE)

### 🎉 Major Achievement
Complete redesign of detection pipeline using **tiled YOLO approach** achieving **99.2% mAP@0.5** - a 2.5× improvement over downscaling approach.

### Added
- **Tiled Dataset Creation** (`scripts/create_tiled_dataset.py`)
  - Tiles 10K×10K images into 1280×1280 patches with 256px overlap
  - Maps ImageJ ROI annotations to tile coordinates
  - Creates YOLO format dataset with train/val/test splits
  - Generated 1,446 tiles from 20 microscope plates

- **Multi-GPU Training** (`scripts/train_yolo_tiled.py`)
  - Distributed training on 4× Quadro RTX 8000 GPUs
  - PyTorch DDP with automatic batch distribution
  - Comprehensive training configuration with augmentations
  - Checkpoint saving every 10 epochs

- **Tiled Inference Pipeline** (`scripts/infer_tiled.py`)
  - Processes ultra-high-resolution images without downscaling
  - Global NMS to merge predictions across tile boundaries
  - Outputs: CSV detections, visualization overlay, metadata JSON
  - Handles 10K×10K images in 2-3 minutes

- **Best Model** (`models/yolo11n_tiled_best.pt`)
  - YOLO11n trained on tiled dataset (epoch 82)
  - Performance: 99.2% mAP@0.5, 97.8% precision, 97.1% recall
  - 2.59M parameters, 6.4 GFLOPs

### Changed
- **Complete pipeline redesign** from SAM-based to YOLO-based detection
- Training approach from full-image downscaling to tiled processing
- Repository structure: archived old approaches, cleaned up active scripts

### Performance Improvements
| Metric | Old (Downscaled) | New (Tiled) | Improvement |
|--------|------------------|-------------|-------------|
| mAP@0.5 | 39.6% | **99.2%** | **+150%** |
| mAP@0.5:0.95 | 16.3% | **85.2%** | **+423%** |
| Precision | 56.4% | **97.8%** | **+73%** |
| Recall | 23.7% | **97.1%** | **+310%** |

### Archived
- Moved old SAM-based and classical CV approaches to `archive_old_scripts/`
- Moved failed training runs to `archive_training_runs/`
- Moved downscaled models to `archive_models/`
- Archived non-tiled dataset to `archive_datasets/`

### Technical Details
- **Dataset**: 14,125 ImageJ ROI annotations from 20 plates
- **Training**: 100 epochs, batch=32, 4 GPUs, ~13 minutes
- **Best Epoch**: 82 (early stopped with patience=30)
- **Tile Strategy**: 1280×1280 with 256px overlap
- **NMS IoU**: 0.5 for merging overlapping detections

### Documentation
- Complete rewrite of `README.md` with tiled pipeline documentation
- Added usage examples, troubleshooting, and performance benchmarks
- Created comprehensive CHANGELOG with milestone summary

---

## [1.0.0] - 2024-12-09 - Initial YOLO Attempts (Archived)

### Added
- ImageJ ROI extraction (`scripts/convert_imagej_rois.py`)
- YOLO dataset preparation from ROIs (`scripts/imagej_rois_to_yolo.py`)
- Initial YOLO training with downscaled images (`scripts/train_yolo_imagej.py`)

### Issues
- Downscaling 10K×10K images to 1280×1280 lost critical details
- Model achieved only 39.6% mAP@0.5
- Very low confidence scores (0.7-0.9%) due to severe information loss
- Approach deemed inadequate and superseded by tiled method

### Archived
- All components moved to archive directories
- Kept for reference only, not recommended for use

---

## [0.x.x] - Pre-2024 - SAM and Classical Methods (Archived)

### Overview
Initial exploration using:
- Segment Anything Model (SAM) with template matching
- Classical watershed segmentation
- Blob detection with size filtering
- Manual annotation-guided approaches

### Location
All SAM and classical CV scripts archived in:
- `archive_old_scripts/` - Template SAM, watershed, blob detection
- `archive_template_approach/` - Full SAM pipeline documentation
- `archive_unused/` - Experimental prototypes

### Reasons for Deprecation
- SAM required extensive manual template curation
- Classical methods had poor recall on cluttered images
- Processing time was prohibitive for large-scale datasets
- YOLO-based approach proved far superior in all metrics

---

## Migration Guide

### From SAM/Classical to Tiled YOLO

**Old workflow**:
```bash
python sam_templates.py image.jpg --template-dir templates/
```

**New workflow**:
```bash
python scripts/infer_tiled.py --image image.jpg --model models/yolo11n_tiled_best.pt
```

**Benefits**:
- 10× faster processing
- 2.5× better accuracy
- No manual template curation needed
- Consistent results across images

### From Downscaled YOLO to Tiled YOLO

**Old approach**: Entire image downscaled to 1280×1280 (88% data loss)  
**New approach**: Image tiled into 1280×1280 patches (0% data loss)

**Migration**: No changes needed, simply use new scripts and model.

---

## Version Numbering

- **Major (X.0.0)**: Complete pipeline redesign or methodology change
- **Minor (x.X.0)**: New features, significant improvements
- **Patch (x.x.X)**: Bug fixes, documentation updates

Current version: **2.0.0**
