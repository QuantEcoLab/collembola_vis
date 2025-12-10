# Collembola Detection & Segmentation

Automated detection, segmentation, and measurement of collembolas (springtails) in microscope images using template-guided Segment Anything Model (SAM).

## Overview

This project uses AI-powered image segmentation to:
- **Detect** collembolas in large microscope images (10K×10K pixels)
- **Segment** individual organisms with high precision
- **Measure** length, width, area, and volume of each specimen
- **Export** results in CSV and JSON formats with visualization overlays

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n collembola python=3.11
conda activate collembola

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segment-anything pandas scikit-image matplotlib tqdm numpy openpyxl
```

### 2. Run Detection

```bash
python sam_templates.py "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --output out/measurements.csv \
    --save-overlay out/overlay.png \
    --auto-download \
    --allow-large-image
```

**The script will automatically:**
- ✓ Download SAM checkpoint (~375MB) if missing
- ✓ Downscale large images (>16MP) to prevent memory issues
- ✓ Subsample templates (max 50) for optimal performance
- ✓ Show progress bars for all operations
- ✓ Generate measurements with volume estimates

## Main Entry Point: `sam_templates.py`

Template-guided SAM segmentation using normalized cross-correlation (NCC) to find candidate regions.

### Key Features

- **Auto-downscaling**: Large images automatically downscaled to prevent hanging
- **Template subsampling**: Reduces 200+ templates to 50 for faster processing
- **Progress feedback**: Clear progress bars and status messages throughout
- **Smart filtering**: Size, aspect ratio, and overlap-based filtering
- **Volume estimation**: Ellipsoid model for 3D volume calculation

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--template-dir` | `data/organism_templates` | Directory with template images |
| `--sam-model-type` | `vit_b` | SAM model: `vit_b` (375MB), `vit_l` (1.2GB), `vit_h` (2.4GB) |
| `--downscale-max-side` | `0` | Max image dimension (auto: 2048 for >16MP images) |
| `--scale-factors` | `1.0` | Template scales (e.g., `0.75,1.0,1.25`) |
| `--ncc-threshold` | `0.6` | NCC correlation threshold (0-1) |
| `--max-prompts` | `400` | Max candidate regions to process |
| `--um-per-pixel` | `8.57` | Microscope calibration (microns/pixel) |
| `--auto-download` | flag | Auto-download SAM checkpoint |
| `--allow-large-image` | flag | Disable PIL decompression warning |

### Output Files

- **CSV** (`--output`): Per-specimen measurements (centroid, length, width, volume)
- **JSON** (`--json`): Complete summary with metadata
- **Overlay** (`--save-overlay`): Visualization with colored masks
- **Masks** (`--save-masks-dir`): Individual binary mask PNGs

## Performance Optimizations

### Before Fixes
- 10408×10338 image (307MB)
- 200 templates × 3 scales = 600 iterations
- **Estimated time: 20+ hours**
- Zero progress feedback → appeared stuck

### After Fixes
- Auto-downscaled to 2048×2033 (96% smaller)
- 50 templates × 1 scale = 50 iterations (92% reduction)
- **Actual time: 5-15 minutes**
- Continuous progress bars and status updates

## Project Structure

```
collembola_vis/
├── sam_templates.py          # Main entry point (latest)
├── volumen.py                 # Volume calculation utilities
├── data/
│   ├── slike/                 # Source microscope images
│   ├── organism_templates/    # Template images for matching
│   └── collembolas_table.csv  # Manual annotations
├── archive_old_scripts/       # Obsolete detection scripts
│   ├── mk_dataset.py          # Dataset creation (blob detection)
│   ├── measure_collembolas.py # Classical watershed method
│   ├── sam_detect.py          # Annotation-guided SAM
│   └── sam_guided.py          # Prototype-based detection
└── checkpoints/               # SAM model weights (auto-downloaded)
```

## Measurement Details

### Units
- **Length/Width**: Micrometers (µm) and millimeters (mm)
- **Area**: Square micrometers (µm²)
- **Volume**: Cubic micrometers (µm³) and cubic millimeters (mm³)

### Volume Calculation
Uses ellipsoid approximation from `volumen.compute_collembola_volume()`:
```python
# Ellipsoid model: V = (4/3) * π * a * b * c
# where a = height/2, b = c = width/2
```

## Troubleshooting

### Script appears stuck
**Solution**: The script now shows progress bars. If truly stuck:
- Reduce templates with existing subsampling (automatic)
- Use smaller `--downscale-max-side` (e.g., 1024)
- Reduce `--scale-factors` to single value: `1.0`

### Out of memory
**Solution**: 
- Increase `--downscale-max-side` reduction (lower value)
- Use smaller SAM model: `--sam-model-type vit_b`

### No detections found
**Solution**:
- Lower `--ncc-threshold` (e.g., 0.5)
- Increase `--scale-factors`: `0.5,0.75,1.0,1.25,1.5`
- Check template quality in `data/organism_templates/`

## Development

### Adding New Templates
Place cropped collembola images in `data/organism_templates/`:
```bash
# Templates should be:
# - Grayscale or RGB (auto-converted)
# - Various sizes (auto-scaled during matching)
# - Clear, well-focused specimens
```

### Calibration
Update `--um-per-pixel` based on microscope settings:
```python
# Calculate from scale bar:
um_per_pixel = scale_bar_length_um / scale_bar_length_pixels
```

## Citation

Project: Collembola Detection & Measurement using SAM  
Repository: https://github.com/QuantEcoLab/collembolae_vis  
Authors: Jana Zovko, Domagoj K. Hackenberger

## License

Research use only. See repository for details.
