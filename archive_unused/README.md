# Archive of Unused Files

**Archived on:** December 8, 2024  
**Reason:** Repository cleanup - moved inactive code and duplicates

## Contents

### `/root_scripts/` - Old Root-Level Scripts
Duplicate scripts that were replaced by the `collembola_pipeline/` module:
- `sam_detect.py` - Old SAM detection script (replaced by `segment.py`)
- `sam_templates.py` - Template-based approach (archived Nov 2024)
- `detect_all.sh`, `detect_auto.sh`, `run_example.sh`, `test_with_overlay.sh` - Shell scripts for old pipeline
- `volumen.py` - Volume calculation (functionality moved to `morphology.py`)

### `/jana_research/` - Research Code
Jana's original research code and experiments:
- `jana_code/` - Original blob detection, cropping, measurement scripts
- Various CSV files and experiment notebooks
- See `jana_code/dz.md` for research notes

### `/documentation/` - Duplicate/Old Documentation
- `readme.md` - Lowercase duplicate (main is `README.md`)
- `ARCHIVE_NOTES.md` - Old archive notes
- `DETECTION_METHODS.md` - Duplicate (exists in `archive_template_approach/`)
- `OVERLAY_GUIDE.md` - Duplicate (exists in `archive_template_approach/`)

### `/data_duplicates/` - Duplicate Data Files
Files that exist in better locations:
- `collembolas_table.csv` → Now in `data/annotations/`
- `crops_dataset.csv` → Now in `data/annotations/`

## Active Codebase

The current active system is:
- **Pipeline:** `collembola_pipeline/` - Modular R-CNN style detection
- **Scripts:** `scripts/` - Training, calibration, validation
- **Data:** `data/slike/`, `data/annotations/`, `data/crops/`
- **Models:** `models/classifier_resnet18.pt`
- **Outputs:** `outputs/csv/`, `outputs/overlays/`

## Restoration

If you need to restore any archived files:
```bash
# Example: restore a script
cp archive_unused/root_scripts/sam_detect.py ./
```

## See Also
- `archive_old_scripts/` - Previously archived measure/dataset scripts
- `archive_template_approach/` - Previously archived template-based detection
