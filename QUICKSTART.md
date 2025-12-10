# Quick Start Guide - Collembola Detection Pipeline

Get started with the tiled YOLO detection pipeline in 5 minutes.

## 🎯 What This Pipeline Does

Detects collembola organisms in ultra-high-resolution microscope images (10K×10K pixels) with **99.2% accuracy**.

**Input**: Large microscope image  
**Output**: CSV with detections, visualization overlay, metadata

## ⚡ Prerequisites

```bash
# Create environment
conda create -n collembola python=3.11
conda activate collembola

# Install dependencies
pip install ultralytics torch pandas pillow numpy read-roi
```

**Hardware**: 
- GPU recommended (NVIDIA with CUDA)
- 8GB+ VRAM for inference
- 32GB+ VRAM per GPU for training

## 🚀 Common Workflows

### 1. Detect Organisms in a New Image

**Use case**: You have a new microscope image and want to detect all collembolas.

```bash
python scripts/infer_tiled.py \
    --image "path/to/your/image.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --conf 0.25 \
    --device 0
```

**Output** (in `infer_tiled_output/`):
- `image_detections.csv` - Bounding boxes with confidence scores
- `image_overlay.jpg` - Visual overlay with red boxes
- `image_metadata.json` - Detection statistics

**Adjust confidence threshold**:
```bash
# More permissive (may include false positives)
--conf 0.15

# More strict (may miss some organisms)
--conf 0.35
```

### 2. Batch Process Multiple Images

```bash
# Create a simple batch script
for image in data/slike/*.jpg; do
    python scripts/infer_tiled.py \
        --image "$image" \
        --model models/yolo11n_tiled_best.pt \
        --output batch_results/
done
```

### 3. Train on Your Own Dataset

**Step 1: Prepare annotations**

Annotate images in ImageJ with ROI manager, export as `.zip` files.

**Step 2: Convert and tile**

```bash
# Extract ROIs to CSV
python scripts/convert_imagej_rois.py

# Create tiled dataset
python scripts/create_tiled_dataset.py
```

**Step 3: Train**

```bash
# Single GPU
python scripts/train_yolo_tiled.py \
    --device 0 \
    --epochs 100 \
    --batch 16

# Multi-GPU (4 GPUs)
python scripts/train_yolo_tiled.py \
    --device 0,1,2,3 \
    --epochs 100 \
    --batch 32
```

**Monitor training**:
- Progress shown in terminal
- Results in `runs/detect/train_*/`
- Best model: `runs/detect/train_*/weights/best.pt`

### 4. Evaluate Model Performance

```bash
# Run validation on test set
from ultralytics import YOLO

model = YOLO('models/yolo11n_tiled_best.pt')
metrics = model.val(data='data/yolo_tiled/data.yaml')

print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall: {metrics.box.r[0]:.3f}")
```

### 5. Export Results to Different Formats

**From CSV to Excel**:
```python
import pandas as pd

df = pd.read_csv('infer_tiled_output/image_detections.csv')
df.to_excel('detections.xlsx', index=False)
```

**From CSV to COCO format**:
```python
import json
import pandas as pd

df = pd.read_csv('infer_tiled_output/image_detections.csv')

coco = {
    "images": [{"id": 1, "file_name": "image.jpg", "width": 10408, "height": 10338}],
    "annotations": [],
    "categories": [{"id": 0, "name": "collembola"}]
}

for idx, row in df.iterrows():
    coco["annotations"].append({
        "id": idx,
        "image_id": 1,
        "category_id": 0,
        "bbox": [row['x1'], row['y1'], row['width'], row['height']],
        "area": row['width'] * row['height'],
        "score": row['confidence']
    })

with open('detections_coco.json', 'w') as f:
    json.dump(coco, f, indent=2)
```

## 🎨 Visualization Tips

### View Overlay in Browser

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('infer_tiled_output/image_overlay.jpg')
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Create Custom Overlay

```python
from PIL import Image, ImageDraw
import pandas as pd

# Load image and detections
img = Image.open('data/slike/image.jpg')
df = pd.read_csv('infer_tiled_output/image_detections.csv')

# Draw boxes
draw = ImageDraw.Draw(img)
for _, row in df.iterrows():
    if row['confidence'] > 0.5:  # High confidence only
        draw.rectangle(
            [row['x1'], row['y1'], row['x2'], row['y2']],
            outline='green',
            width=5
        )

img.save('custom_overlay.jpg', quality=95)
```

## 📊 Understanding Output

### CSV Columns

| Column | Description | Example |
|--------|-------------|---------|
| `x1, y1` | Top-left corner (pixels) | 7440.3, 2143.5 |
| `x2, y2` | Bottom-right corner (pixels) | 7612.7, 2295.6 |
| `width, height` | Box dimensions (pixels) | 172.4, 152.1 |
| `confidence` | Detection confidence (0-1) | 0.961 |
| `class` | Object class (always 0 for collembola) | 0 |

### Metadata JSON

```json
{
  "image_size": [10408, 10338],      // Original image dimensions
  "num_tiles": 100,                  // Number of tiles processed
  "detections_before_nms": 1214,     // Raw detections
  "detections_after_nms": 800,       // After duplicate removal
  "conf_threshold": 0.25,            // Confidence cutoff used
  "iou_threshold": 0.5               // NMS overlap threshold
}
```

## 🔧 Performance Tuning

### Inference Speed

**Faster inference** (lower quality):
```bash
python scripts/infer_tiled.py \
    --conf 0.35 \      # Higher threshold = fewer detections to process
    --iou 0.6          # More aggressive NMS
```

**Better accuracy** (slower):
```bash
python scripts/infer_tiled.py \
    --conf 0.15 \      # Lower threshold = catch more organisms
    --iou 0.4          # Less aggressive NMS
```

### Memory Management

**Out of memory?**
```bash
# Use CPU (slow but always works)
python scripts/infer_tiled.py --device cpu

# Or reduce batch size during training
python scripts/train_yolo_tiled.py --batch 8
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **"No detections found"** | Lower `--conf` to 0.15 or check if image format is supported |
| **"CUDA out of memory"** | Use `--device cpu` or smaller batch size |
| **"Model not found"** | Check path: `models/yolo11n_tiled_best.pt` exists |
| **"Too many false positives"** | Increase `--conf` to 0.35 or 0.40 |
| **"Missing some organisms"** | Lower `--conf` to 0.15 or retrain with more data |

## 📁 File Locations

```
collembola_vis/
├── scripts/
│   ├── infer_tiled.py              # ← Main inference script
│   ├── train_yolo_tiled.py         # ← Training script
│   └── create_tiled_dataset.py     # ← Dataset preparation
├── models/
│   └── yolo11n_tiled_best.pt       # ← Pre-trained model (99.2% mAP)
├── data/
│   ├── slike/                      # ← Put your images here
│   └── yolo_tiled/                 # ← Training dataset
├── infer_tiled_output/             # ← Results appear here
└── runs/detect/                    # ← Training runs
```

## 🎓 Next Steps

1. **Try the pipeline**: Run inference on a sample image
2. **Review results**: Check CSV and overlay for accuracy
3. **Adjust parameters**: Tune confidence threshold for your use case
4. **Train custom model**: If needed, annotate your own data
5. **Integrate into workflow**: Batch process, export to database, etc.

## 💡 Pro Tips

- **Batch processing**: Use a shell loop or Python script to process many images
- **Quality check**: Always review overlays for first few images to validate thresholds
- **Backup models**: Keep `best.pt` from training runs in case you need to revert
- **Version control**: Track which model version was used for each analysis
- **Calibration**: Store microscope calibration (µm/pixel) in metadata for measurements

## 📚 More Information

- **Full documentation**: See [README.md](README.md)
- **Training details**: See [CHANGELOG.md](CHANGELOG.md)
- **Repository guidelines**: See [AGENTS.md](AGENTS.md)

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review full [README.md](README.md) documentation
3. Check if issue is mentioned in [CHANGELOG.md](CHANGELOG.md)
4. Look at archived approaches in `archive_documentation/` for context

---

**Ready?** Start with a simple inference:

```bash
conda activate collembola
python scripts/infer_tiled.py \
    --image "data/slike/K1_Fe2O3001 (1).jpg" \
    --model models/yolo11n_tiled_best.pt
```

Then check `infer_tiled_output/` for results!
