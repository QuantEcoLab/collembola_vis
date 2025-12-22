# Upload Instructions for YOLO11n Collembola Detection Model

## Files Prepared for Upload

### Model Files
- `models/yolo11n_tiled_best.pt` (5.4 MB) - Main model checkpoint

### Documentation
- `MODEL_CARD.md` - Comprehensive model documentation
- `HUGGINGFACE_README.md` - Hugging Face Hub README
- `.zenodo.json` - Zenodo metadata
- `README.md` - Full pipeline documentation
- `PERFORMANCE.md` - Performance benchmarks
- `QUICKSTART.md` - Quick start guide

### Example Scripts
- `scripts/infer_tiled.py` - Tiled inference script
- `scripts/measure_organisms_fast.py` - Fast measurement script
- `scripts/process_plate_batch.py` - Batch processing

## 📦 Zenodo Upload

### Step 1: Create Upload Package

```bash
# Create upload directory
mkdir -p zenodo_upload

# Copy model and documentation
cp models/yolo11n_tiled_best.pt zenodo_upload/
cp MODEL_CARD.md zenodo_upload/
cp README.md zenodo_upload/
cp PERFORMANCE.md zenodo_upload/
cp QUICKSTART.md zenodo_upload/
cp .zenodo.json zenodo_upload/

# Copy example scripts
mkdir -p zenodo_upload/scripts
cp scripts/infer_tiled.py zenodo_upload/scripts/
cp scripts/measure_organisms_fast.py zenodo_upload/scripts/
cp scripts/process_plate_batch.py zenodo_upload/scripts/

# Create archive
cd zenodo_upload
zip -r ../collembola_yolo11n_model_v1.0.0.zip .
cd ..
```

### Step 2: Upload to Zenodo

1. Go to https://zenodo.org/
2. Click "New upload"
3. Upload `collembola_yolo11n_model_v1.0.0.zip`
4. Zenodo will read `.zenodo.json` for metadata
5. Review and publish

### Zenodo Metadata

The `.zenodo.json` file contains:
- **Title**: YOLO11n Collembola Detection Model - Tiled Inference for Ultra-High-Resolution Microscope Images
- **Creators**: Jana Zovko, Domagoj K. Hackenberger
- **License**: CC-BY-NC-4.0
- **Keywords**: YOLO11, object detection, collembola, microscopy, deep learning
- **Version**: 1.0.0

## 🤗 Hugging Face Hub Upload

### Step 1: Install Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Create Repository

```bash
# Create new model repository
huggingface-cli repo create collembola-yolo11n --type model

# Or use Python API
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('collembola-yolo11n', repo_type='model', exist_ok=True)
"
```

### Step 3: Upload Files

```bash
# Upload model and documentation
huggingface-cli upload QuantEcoLab/collembola-yolo11n models/yolo11n_tiled_best.pt yolo11n_tiled_best.pt
huggingface-cli upload QuantEcoLab/collembola-yolo11n HUGGINGFACE_README.md README.md
huggingface-cli upload QuantEcoLab/collembola-yolo11n MODEL_CARD.md MODEL_CARD.md
huggingface-cli upload QuantEcoLab/collembola-yolo11n PERFORMANCE.md PERFORMANCE.md

# Or use Python API
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Upload model
api.upload_file(
    path_or_fileobj='models/yolo11n_tiled_best.pt',
    path_in_repo='yolo11n_tiled_best.pt',
    repo_id='QuantEcoLab/collembola-yolo11n',
    repo_type='model'
)

# Upload README (will be displayed on model page)
api.upload_file(
    path_or_fileobj='HUGGINGFACE_README.md',
    path_in_repo='README.md',
    repo_id='QuantEcoLab/collembola-yolo11n',
    repo_type='model'
)
"
```

### Step 4: Test Download

```bash
# Test download from Hugging Face
python -c "
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id='QuantEcoLab/collembola-yolo11n',
    filename='yolo11n_tiled_best.pt'
)
print(f'Model downloaded to: {model_path}')

# Test inference
from ultralytics import YOLO
model = YOLO(model_path)
print('Model loaded successfully!')
"
```

## 📋 Upload Checklist

### Before Upload

- [x] Model file ready: `models/yolo11n_tiled_best.pt` (5.4 MB)
- [x] Documentation created: `MODEL_CARD.md`
- [x] Zenodo metadata: `.zenodo.json`
- [x] Hugging Face README: `HUGGINGFACE_README.md`
- [ ] Test model loading locally
- [ ] Verify all scripts work with uploaded model

### Zenodo Upload

- [ ] Create Zenodo account / login
- [ ] Create ZIP package
- [ ] Upload to Zenodo
- [ ] Review auto-filled metadata
- [ ] Publish
- [ ] Note DOI

### Hugging Face Upload

- [ ] Create Hugging Face account / login
- [ ] Install `huggingface_hub` CLI
- [ ] Create model repository
- [ ] Upload model file
- [ ] Upload README and documentation
- [ ] Set license (CC-BY-NC-4.0)
- [ ] Add tags: yolo11, object-detection, collembola, microscopy
- [ ] Test download

## 🔗 Repository Links

After upload, update these in documentation:

### Zenodo
```
DOI: 10.5281/zenodo.XXXXXXX
URL: https://zenodo.org/record/XXXXXXX
```

### Hugging Face
```
Model Hub: https://huggingface.co/QuantEcoLab/collembola-yolo11n
```

### GitHub
```
Source Code: https://github.com/QuantEcoLab/collembolae_vis
```

## 📝 Usage Examples

### From Zenodo

```bash
# Download from Zenodo
wget https://zenodo.org/record/XXXXXXX/files/collembola_yolo11n_model_v1.0.0.zip
unzip collembola_yolo11n_model_v1.0.0.zip
cd collembola_yolo11n_model_v1.0.0

# Run inference
python scripts/infer_tiled.py --image image.jpg --model yolo11n_tiled_best.pt
```

### From Hugging Face

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download model
model_path = hf_hub_download(
    repo_id='QuantEcoLab/collembola-yolo11n',
    filename='yolo11n_tiled_best.pt'
)

# Load and use
model = YOLO(model_path)
results = model('image.jpg', conf=0.6)
```

## 🎯 Post-Upload Tasks

1. **Update README.md** - Add Zenodo DOI badge and Hugging Face link
2. **Update GitHub Release** - Create v1.0.0 release with model
3. **Update Citation** - Add DOI to bibtex citation
4. **Documentation** - Reference model locations in docs
5. **Test** - Verify downloads work from both platforms

## 📧 Support

For issues or questions:
- **GitHub Issues**: https://github.com/QuantEcoLab/collembolae_vis/issues
- **Contact**: [Your email address]
