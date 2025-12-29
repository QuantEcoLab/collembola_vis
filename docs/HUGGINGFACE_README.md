---
license: cc-by-nc-4.0
tags:
- yolo11
- object-detection
- collembola
- microscopy
- ecology
- computer-vision
- ultralytics
library_name: ultralytics
pipeline_tag: object-detection
---

# YOLO11n Collembola Detection Model

## Model Description

A **YOLO11n** model for detecting collembola organisms in **ultra-high-resolution microscope images** (10K×10K pixels). Uses tiled inference to maintain detection accuracy without downscaling.

- **Model**: YOLO11n (nano variant)
- **Parameters**: 2.59M
- **Model Size**: 5.4 MB
- **Input**: 1280×1280 pixels (per tile)
- **mAP@0.5**: **99.2%**
- **Precision**: **97.8%**
- **Recall**: **97.1%**

## Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n_tiled_best.pt')

# Simple inference (for images ≤1280×1280)
results = model('image.jpg', conf=0.6)

# For larger images, use tiled inference (see MODEL_CARD.md)
```

## Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 99.2% |
| mAP@0.5:0.95 | 85.2% |
| Precision | 97.8% |
| Recall | 97.1% |
| Inference Speed | ~2-3 min (10K×10K image, single GPU) |

## Training Data

- **Annotations**: 14,125 ImageJ ROI from 20 microscope plates
- **Tiled Dataset**: 1,246 training tiles (16,701 annotations)
- **Validation**: 200 tiles (3,950 annotations)
- **Image Resolution**: 10408×10338 pixels (~108 MP)

## Why Tiled Inference?

| Approach | mAP@0.5 | Precision | Recall |
|----------|---------|-----------|--------|
| Downscaled (10K→1280) | 39.6% | 56.4% | 23.7% |
| **Tiled (1280 patches)** | **99.2%** | **97.8%** | **97.1%** |

**Tiled inference is 2.5× better** - no detail loss!

## Usage

### Installation

```bash
pip install ultralytics torch pillow numpy
```

### Tiled Inference

```python
from ultralytics import YOLO
import cv2
import torch
from torchvision.ops import nms

def detect_tiled(image_path, model, tile_size=1280, overlap=256, conf=0.6):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    stride = tile_size - overlap
    detections = []
    
    # Process tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]
            
            results = model(tile, conf=conf, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                detections.append([
                    x1 + x, y1 + y, x2_box + x, y2_box + y,
                    float(box.conf[0]), int(box.cls[0])
                ])
    
    # Global NMS
    if detections:
        boxes = torch.tensor([d[:4] for d in detections])
        scores = torch.tensor([d[4] for d in detections])
        keep = nms(boxes, scores, iou_threshold=0.5)
        detections = [detections[i] for i in keep]
    
    return detections

# Run detection
model = YOLO('yolo11n_tiled_best.pt')
detections = detect_tiled('image.jpg', model, conf=0.6)
print(f"Found {len(detections)} organisms")
```

### Full Pipeline

For complete detection + measurement pipeline, see the [GitHub repository](https://github.com/QuantEcoLab/collembolae_vis):

```bash
git clone https://github.com/QuantEcoLab/collembolae_vis.git
cd collembolae_vis

# Run tiled detection
python scripts/infer_tiled.py \
    --image "data/slike/image.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --conf 0.6 \
    --device 0
```

## Model Details

### Classes
- **Class 0**: Collembola body (without antennae and furca)

### Tiling Configuration
- **Tile Size**: 1280×1280 pixels
- **Overlap**: 256 pixels (20%)
- **Stride**: 1024 pixels
- **Global NMS**: IoU threshold 0.5

### Training Configuration
- **Hardware**: 4× Quadro RTX 8000 (48GB VRAM)
- **Batch Size**: 32 (8 per GPU)
- **Epochs**: 100 (best at epoch 82)
- **Optimizer**: AdamW
- **Augmentation**: Mosaic, flip, HSV, scale

## Limitations

- Optimized for microscope images at similar resolution/magnification
- Best with confidence threshold ≥ 0.6
- Requires tiled inference for images > 1280×1280

## Citation

```bibtex
@software{collembola_yolo_2024,
  title = {YOLO11n Collembola Detection Model},
  author = {Zovko, Jana and Hackenberger, Domagoj K.},
  year = {2024},
  url = {https://github.com/QuantEcoLab/collembolae_vis}
}
```

## Links

- **GitHub**: https://github.com/QuantEcoLab/collembolae_vis
- **Documentation**: Full pipeline docs in repository
- **License**: CC BY-NC 4.0 (Research use only)
