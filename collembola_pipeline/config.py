from pathlib import Path

# Paths
DATA_DIR = Path("data")
PLATES_DIR = DATA_DIR / "slike"
CROPS_DIR = DATA_DIR / "crops"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CALIBRATION_DIR = DATA_DIR / "calibration"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
CSV_DIR = OUTPUTS_DIR / "csv"
OVERLAYS_DIR = OUTPUTS_DIR / "overlays"

# Calibration (to be set after measuring)
PX_TO_UM: float = 1.0  # placeholder, update via scripts/01_calibrate.py

# Classifier
CLASSIFIER_PATH = MODELS_DIR / "classifier_resnet18.pt"
# Optimized based on K1 TP/FP analysis: removes 44% of FPs while keeping 89% recall
# Previous: 0.9 (P=42.7%, R=79.2%, F1=55.5%)
# New: 0.99 (P=54.2%, R=89.1%, F1=67.4% when combined with eccentricity filter)
CLASSIFIER_THRESHOLD: float = 0.99

# SAM
SAM_CHECKPOINT = Path("checkpoints/sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"
SAM_AUTOMASK_PARAMS = dict(
    points_per_side=32,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.9,
    min_mask_region_area=20,
    box_nms_thresh=0.7,
)

# Optional per-device overrides for SAM AutoMask to boost recall
SAM_AUTOMASK_GPU_PARAMS = dict(
    points_per_side=32,
    points_per_batch=16,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.85,
    min_mask_region_area=10,
    crop_n_layers=0,
    crop_overlap_ratio=0.5,
    crop_n_points_downscale_factor=2,
    box_nms_thresh=0.7,
)
SAM_AUTOMASK_CPU_PARAMS = dict(
    points_per_side=48,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.85,
    min_mask_region_area=10,
    crop_n_layers=2,
    crop_overlap_ratio=0.5,
    crop_n_points_downscale_factor=2,
    box_nms_thresh=0.7,
)


# Device defaults
DEFAULT_DEVICE = "cuda"  # prefer GPU when available

# Downscaling
# When running on CPU, large 10k×10k images can OOM; auto-downscale to this max side.
DOWNSCALE_MAX_SIDE: int = 1024

# GPU tiling for SAM on very large plates
TILE_MAX_SIDE: int = 2304  # process larger images in overlapping tiles on GPU
TILE_OVERLAP: int = 192  # pixels of overlap between tiles

# Post-classification morphology filters (tunable)
MIN_AREA_PX: int = 200
MIN_MAJOR_PX: int = 40
MIN_MINOR_PX: int = 6
MAX_AREA_PX: int = 20000
MAX_MAJOR_PX: int = 600
MAX_MINOR_PX: int = 250
# Shape filters (precision-oriented)
# Optimized based on K1 TP/FP analysis:
# - MIN_ECCENTRICITY increased from 0.70 to 0.89 (removes rounder false positives)
# - Targets 89% recall with 54% precision (F1=67.4%)
MIN_SOLIDITY: float = 0.60  # Keep conservative (TP min=0.632)
MIN_ECCENTRICITY: float = 0.89  # Data-driven: removes 44% FPs, keeps 89% recall
MIN_ASPECT_RATIO: float = 0.08  # minor/major
MAX_ASPECT_RATIO: float = 0.65

# Overlay labeling
OVERLAY_MIN_LABEL_CONF: float = 0.90
