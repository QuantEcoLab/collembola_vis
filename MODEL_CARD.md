# YOLO11n Collembola Detection Model

## Model Description

A YOLO11n model trained for detecting collembola organisms in ultra-high-resolution microscope images (10K×10K pixels). The model uses a tiled inference approach with 1280×1280 pixel patches to maintain detection accuracy without image downscaling.

**Model Name**: `yolo11n_tiled_best.pt`  
**Framework**: Ultralytics YOLO11  
**Architecture**: YOLO11n (nano variant)  
**Parameters**: 2.59M  
**Model Size**: 5.4 MB  
**Input Size**: 1280×1280 pixels (per tile)

## Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **99.2%** |
| **mAP@0.5:0.95** | **85.2%** |
| **Precision** | **97.8%** |
| **Recall** | **97.1%** |

**Inference Speed**: ~2-3 minutes for a 10K×10K image on a single GPU

## Training Data

- **Source**: 14,125 ImageJ ROI annotations from 20 microscope plates
- **Annotation Type**: "bez antene i furce" (collembola bodies without antennae and furca)
- **Tiled Dataset**: 
  - Training: 1,246 tiles (16,701 annotations)
  - Validation: 200 tiles (3,950 annotations)
- **Tile Configuration**: 1280×1280 with 256px overlap
- **Original Image Resolution**: 10408×10338 pixels (~108 megapixels)

## Training Configuration

- **Model**: YOLO11n (2.59M parameters)
- **Training Hardware**: 4× Quadro RTX 8000 (48GB VRAM each)
- **Batch Size**: 32 (8 per GPU)
- **Epochs**: 100 (best at epoch 82)
- **Optimizer**: AdamW (auto-selected)
- **Image Size**: 1280×1280
- **Data Augmentation**: Mosaic, flip, HSV, scale
- **Early Stopping**: Patience 30 epochs

## Usage

### Installation

```bash
pip install ultralytics torch pillow numpy
```

### Inference on Full Image

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO('yolo11n_tiled_best.pt')

# Tiled inference function
def detect_tiled(image_path, model, tile_size=1280, overlap=256, conf=0.6):
    """
    Run tiled YOLO detection on ultra-high-resolution image.
    
    Args:
        image_path: Path to input image
        model: YOLO model instance
        tile_size: Size of each tile (default: 1280)
        overlap: Overlap between tiles (default: 256)
        conf: Confidence threshold (default: 0.6)
    
    Returns:
        List of detections [x1, y1, x2, y2, confidence, class]
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    stride = tile_size - overlap
    detections = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]
            
            # Run detection
            results = model(tile, conf=conf, verbose=False)
            
            # Convert to full image coordinates
            for box in results[0].boxes:
                x1, y1, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                detections.append([
                    x1 + x, y1 + y, x2_box + x, y2_box + y,
                    float(box.conf[0]), int(box.cls[0])
                ])
    
    # Apply global NMS
    from torchvision.ops import nms
    import torch
    
    if len(detections) > 0:
        boxes = torch.tensor([d[:4] for d in detections])
        scores = torch.tensor([d[4] for d in detections])
        keep = nms(boxes, scores, iou_threshold=0.5)
        detections = [detections[i] for i in keep]
    
    return detections

# Run detection
detections = detect_tiled('path/to/image.jpg', model, conf=0.6)
print(f"Found {len(detections)} organisms")
```

### Using the Pipeline Scripts

```bash
# Clone repository
git clone https://github.com/QuantEcoLab/collembolae_vis.git
cd collembolae_vis

# Run detection
python scripts/infer_tiled.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --model models/yolo11n_tiled_best.pt \
    --conf 0.6 \
    --device 0
```

## Tiling Strategy

The model is designed for tiled inference to handle ultra-high-resolution images:

- **Tile Size**: 1280×1280 pixels
- **Overlap**: 256 pixels (20%)
- **Stride**: 1024 pixels
- **Global NMS**: IoU threshold 0.5 to merge overlapping detections

This approach is **2.5× better** than downscaling (99.2% vs 39.6% mAP@0.5) while preserving all image details.

## Classes

- **Class 0**: Collembola body (without antennae and furca)

## Limitations

- Model is optimized for microscope images at similar resolution and magnification
- Requires tiled inference for images larger than 1280×1280
- Best performance with confidence threshold ≥ 0.6 to reduce false positives

## Citation

```bibtex
@software{collembola_yolo_2024,
  title = {YOLO11n Collembola Detection Model},
  author = {Zovko, Jana and Hackenberger, Domagoj K.},
  year = {2024},
  url = {https://github.com/QuantEcoLab/collembolae_vis},
  note = {Trained on 14,125 ImageJ ROI annotations from 20 microscope plates}
}
```

## License

Research use only. See repository for details.

## Repository

**GitHub**: https://github.com/QuantEcoLab/collembolae_vis  
**Documentation**: See `README.md` for full pipeline documentation  
**Performance**: See `PERFORMANCE.md` for benchmarks
