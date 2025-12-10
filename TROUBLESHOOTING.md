# Troubleshooting Guide - Collembola Detection Pipeline

Common issues and solutions for the tiled YOLO detection pipeline.

## 🔍 Inference Issues

### No Detections Found

**Symptom**: `infer_tiled.py` outputs "0 detections" or very few detections.

**Possible Causes & Solutions**:

1. **Confidence threshold too high**
   ```bash
   # Try lower threshold (default is 0.25)
   python scripts/infer_tiled.py --image image.jpg --conf 0.15
   
   # Or even lower for very small organisms
   python scripts/infer_tiled.py --image image.jpg --conf 0.10
   ```

2. **Wrong model loaded**
   ```bash
   # Verify model path
   ls -lh models/yolo11n_tiled_best.pt
   
   # Use explicit path
   python scripts/infer_tiled.py \
       --image image.jpg \
       --model models/yolo11n_tiled_best.pt
   ```

3. **Image format not supported**
   ```python
   from PIL import Image
   
   # Check if image can be opened
   img = Image.open('your_image.jpg')
   print(f"Size: {img.size}, Mode: {img.mode}")
   
   # Convert if needed
   img = img.convert('RGB')
   img.save('converted.jpg')
   ```

4. **Image too different from training data**
   - Check if magnification/microscope settings match training images
   - Organism size should be in expected range (20-200 pixels)
   - Consider retraining with your specific images

### Too Many False Positives

**Symptom**: Many boxes around dirt, debris, or artifacts.

**Solutions**:

1. **Increase confidence threshold**
   ```bash
   # Be more selective (default: 0.25)
   python scripts/infer_tiled.py --image image.jpg --conf 0.35
   
   # Very strict
   python scripts/infer_tiled.py --image image.jpg --conf 0.50
   ```

2. **Adjust NMS threshold**
   ```bash
   # More aggressive duplicate removal
   python scripts/infer_tiled.py --image image.jpg --iou 0.3
   ```

3. **Post-process detections**
   ```python
   import pandas as pd
   
   df = pd.read_csv('infer_tiled_output/image_detections.csv')
   
   # Filter by confidence
   df_filtered = df[df['confidence'] > 0.4]
   
   # Filter by size (adjust based on expected organism size)
   df_filtered = df_filtered[
       (df_filtered['width'] > 30) & 
       (df_filtered['width'] < 300) &
       (df_filtered['height'] > 30) &
       (df_filtered['height'] < 300)
   ]
   
   df_filtered.to_csv('filtered_detections.csv', index=False)
   ```

### Missing Some Organisms

**Symptom**: Known organisms not detected.

**Solutions**:

1. **Lower confidence threshold**
   ```bash
   python scripts/infer_tiled.py --image image.jpg --conf 0.15
   ```

2. **Check organism characteristics**
   - Are they very small? May need to retrain with smaller objects
   - Are they overlapping? NMS may be removing valid detections
   - Are they at tile edges? Check tile overlap settings

3. **Adjust NMS for overlapping organisms**
   ```bash
   # Less aggressive NMS (allows more overlap)
   python scripts/infer_tiled.py --image image.jpg --iou 0.6
   ```

4. **Manually inspect missed organisms**
   ```python
   from PIL import Image, ImageDraw
   import pandas as pd
   
   # Load image and detections
   img = Image.open('image.jpg')
   df = pd.read_csv('infer_tiled_output/image_detections.csv')
   
   # Draw all detections with confidence scores
   draw = ImageDraw.Draw(img)
   for _, row in df.iterrows():
       color = 'green' if row['confidence'] > 0.25 else 'yellow'
       draw.rectangle([row['x1'], row['y1'], row['x2'], row['y2']], 
                     outline=color, width=3)
       draw.text((row['x1'], row['y1']-15), 
                f"{row['confidence']:.2f}", fill=color)
   
   img.save('all_detections_with_conf.jpg')
   ```

## 💾 Memory Issues

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory` during inference or training.

**Solutions**:

1. **Use CPU instead of GPU** (slower but works)
   ```bash
   python scripts/infer_tiled.py --image image.jpg --device cpu
   ```

2. **Clear GPU cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Reduce batch size during training**
   ```bash
   # Default: 32 for 4 GPUs
   python scripts/train_yolo_tiled.py --batch 16 --device 0,1,2,3
   
   # For single GPU
   python scripts/train_yolo_tiled.py --batch 8 --device 0
   ```

4. **Use smaller image size** (requires retraining)
   ```bash
   # Train with 640 instead of 1280
   python scripts/train_yolo_tiled.py --imgsz 640
   ```

5. **Monitor GPU usage**
   ```bash
   # In separate terminal
   watch -n 1 nvidia-smi
   ```

### PIL DecompressionBomb Warning

**Symptom**: `PIL.Image.DecompressionBombWarning` for large images.

**Solution**: This is just a warning and can be ignored for large microscope images.

```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable warning
```

Or in script:
```bash
export PYTHONWARNINGS="ignore::PIL.Image.DecompressionBombWarning"
python scripts/infer_tiled.py --image large_image.jpg
```

## 🏋️ Training Issues

### Training Not Starting

**Symptom**: Script hangs or shows no progress.

**Checks**:

1. **Verify dataset exists**
   ```bash
   ls -lh data/yolo_tiled/
   # Should show: images/, labels/, data.yaml
   
   cat data/yolo_tiled/data.yaml
   # Verify paths are correct
   ```

2. **Check GPU availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

3. **Verify batch size is compatible with GPU count**
   ```bash
   # For 4 GPUs, batch must be multiple of 4
   python scripts/train_yolo_tiled.py --batch 32 --device 0,1,2,3  # ✓ OK
   python scripts/train_yolo_tiled.py --batch 30 --device 0,1,2,3  # ✗ Bad
   ```

### Poor Training Performance

**Symptom**: Low mAP (<50%) after many epochs.

**Possible Causes**:

1. **Insufficient training data**
   - Need at least 500+ annotations
   - Check: `cat data/yolo_tiled/data.yaml`

2. **Class imbalance**
   ```bash
   # Count annotations
   wc -l data/yolo_tiled/labels/train/*.txt
   
   # Check if annotations are distributed across images
   ```

3. **Learning rate too high/low**
   ```bash
   # Try different learning rates
   python scripts/train_yolo_tiled.py --lr0 0.005  # Lower
   python scripts/train_yolo_tiled.py --lr0 0.02   # Higher
   ```

4. **Wrong data augmentation**
   - Check `train_yolo_tiled.py` augmentation settings
   - May need to adjust for your specific images

### Training Crashes

**Symptom**: Training stops with error or system freeze.

**Solutions**:

1. **Multi-GPU issues**
   ```bash
   # Try single GPU first
   python scripts/train_yolo_tiled.py --batch 16 --device 0
   
   # Check if DDP works
   export NCCL_DEBUG=INFO  # Debug distributed training
   python scripts/train_yolo_tiled.py --batch 32 --device 0,1
   ```

2. **Corrupted images**
   ```python
   from PIL import Image
   import os
   
   # Check all training images
   for img_path in os.listdir('data/yolo_tiled/images/train/'):
       try:
           img = Image.open(f'data/yolo_tiled/images/train/{img_path}')
           img.verify()
       except Exception as e:
           print(f"Corrupted: {img_path} - {e}")
   ```

3. **Disk space issues**
   ```bash
   df -h  # Check available space
   # Training checkpoints can use significant space
   ```

## 📊 Dataset Issues

### ROI Extraction Fails

**Symptom**: `convert_imagej_rois.py` produces no output or errors.

**Solutions**:

1. **Check ROI file format**
   ```bash
   # ROI files should be .zip containing ImageJ ROIs
   unzip -l "path/to/RoiSet.zip"
   ```

2. **Verify read-roi installation**
   ```bash
   pip install read-roi
   python -c "import read_roi; print(read_roi.__version__)"
   ```

3. **Check file paths**
   ```python
   import os
   
   # Verify paths in convert_imagej_rois.py
   roi_dir = "data/training_data/Collembola_ROI setovi/..."
   print(f"Exists: {os.path.exists(roi_dir)}")
   print(f"Contents: {os.listdir(roi_dir)}")
   ```

### Tiled Dataset Creation Issues

**Symptom**: `create_tiled_dataset.py` creates no tiles or wrong number.

**Checks**:

1. **Verify input paths**
   ```python
   # Check paths in create_tiled_dataset.py
   ROI_CSV = "data/annotations/imagej_rois_bez.csv"
   IMAGE_BASE = "data/training_data/..."
   ```

2. **Check ROI CSV format**
   ```bash
   head data/annotations/imagej_rois_bez.csv
   # Should have columns: plate_id,image_name,x,y,w,h,roi_name,roi_type
   ```

3. **Monitor tile creation**
   ```bash
   # Watch tile directory while script runs
   watch -n 1 "ls data/yolo_tiled/images/train/ | wc -l"
   ```

4. **Check for "empty" tiles (no annotations)**
   - Script only saves tiles with annotations by default
   - To save all tiles (including background), edit `create_tiled_dataset.py`:
     ```python
     # Change this line:
     if len(yolo_annotations) > 0:  # Only save tiles with annotations
     # To:
     # Always save (for background learning)
     ```

## 🖼️ Visualization Issues

### Overlay Not Showing Boxes

**Symptom**: Overlay image created but no boxes visible.

**Solutions**:

1. **Check detection count**
   ```bash
   wc -l infer_tiled_output/*_detections.csv
   # Should be > 1 (header + detections)
   ```

2. **Verify box coordinates**
   ```python
   import pandas as pd
   df = pd.read_csv('infer_tiled_output/image_detections.csv')
   print(df.head())
   
   # Check if coordinates are within image bounds
   print(f"x1 range: {df['x1'].min()} - {df['x1'].max()}")
   print(f"y1 range: {df['y1'].min()} - {df['y1'].max()}")
   ```

3. **Boxes may be too small to see**
   - Zoom into overlay image with image viewer
   - Increase box width in `infer_tiled.py`:
     ```python
     draw.rectangle([x1, y1, x2, y2], outline='red', width=5)  # Increase width
     ```

### Can't Open Overlay File

**Symptom**: Overlay .jpg file is corrupted or won't open.

**Solutions**:

1. **Check file size**
   ```bash
   ls -lh infer_tiled_output/*_overlay.jpg
   # Should be several MB for 10K×10K image
   ```

2. **Re-create overlay manually**
   ```python
   from PIL import Image, ImageDraw
   import pandas as pd
   
   img = Image.open('original_image.jpg')
   df = pd.read_csv('infer_tiled_output/image_detections.csv')
   
   draw = ImageDraw.Draw(img)
   for _, row in df.iterrows():
       draw.rectangle([row['x1'], row['y1'], row['x2'], row['y2']], 
                     outline='red', width=3)
   
   img.save('manual_overlay.jpg', quality=95)
   ```

## 🔧 Environment Issues

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
# Activate environment
conda activate collembola

# Install missing packages
pip install ultralytics torch pandas pillow numpy read-roi

# Verify installation
python -c "import ultralytics; print(ultralytics.__version__)"
```

### CUDA/PyTorch Mismatch

**Symptom**: `RuntimeError: CUDA error` or `torch.cuda.is_available()` returns False.

**Solution**:
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check system CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📝 Output Format Issues

### CSV Encoding Problems

**Symptom**: Special characters in filenames cause CSV read errors.

**Solution**:
```python
import pandas as pd

# Specify encoding when reading
df = pd.read_csv('detections.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('detections.csv', encoding='latin1')
```

### JSON Parsing Errors

**Symptom**: `json.decoder.JSONDecodeError` when reading metadata.

**Solution**:
```python
import json

# Check if file is valid JSON
with open('metadata.json', 'r') as f:
    content = f.read()
    print(content)  # Look for syntax errors

# Try parsing
try:
    data = json.loads(content)
except json.JSONDecodeError as e:
    print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
```

## 🆘 Still Having Issues?

1. **Check logs**: Look in `runs/detect/train_*/` for training logs
2. **Review code**: Scripts have detailed comments explaining each step
3. **Test with sample data**: Try inference on known-good images first
4. **Check system resources**: Monitor CPU, GPU, RAM, disk space
5. **Simplify**: Strip down to minimal command and add parameters incrementally

### Useful Debug Commands

```bash
# System info
nvidia-smi
python --version
conda list

# Check data
ls -lh data/yolo_tiled/
head data/yolo_tiled/data.yaml

# Test imports
python -c "import ultralytics, torch, pandas, PIL; print('OK')"

# GPU test
python -c "import torch; print(torch.cuda.is_available())"

# Model test
python -c "from ultralytics import YOLO; m=YOLO('models/yolo11n_tiled_best.pt'); print(m)"
```

### Log Locations

- **Training logs**: `runs/detect/train_*/train_*.log`
- **Results CSV**: `runs/detect/train_*/results.csv`
- **Validation metrics**: `runs/detect/train_*/val/`
- **Inference output**: `infer_tiled_output/*_metadata.json`

---

**Can't find your issue?** Check the full [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md) for more details.
