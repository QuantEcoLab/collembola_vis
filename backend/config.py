"""Central configuration for the web backend."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with COLLEMBOLA_ env prefix."""

    # Paths (relative to repo root)
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    outputs_dir: Path = Path("data/web_outputs")
    calibration_dir: Path = Path("data/calibration")
    annotations_dir: Path = Path("data/annotations")
    models_dir: Path = Path("models")
    detector_dataset_dir: Path = Path("data/curated_detector")
    detector_split_registry: Path = Path("data/detector_split_registry.json")

    # Standard detection profile. The detector is the production model fine-tuned on reviewed and
    # visually verified Pendifin (CAC/ERD) train plates with hard negatives (v2); it runs on 1280 tiles,
    # followed by global NMS and suppression of fragment boxes lying >= default_contain inside a larger
    # box. Optional extra tile passes (e.g. 640 px tiles seen 2x upscaled) are supported but off by
    # default. The weights are not in git: install them at default_model before deploying; startup
    # refuses a missing file or another hash (see docs/DETECTION_RECALL.md).
    default_model: Path = Path("models/yolo11n_tiled_pendifin_v2.pt")
    # Checked at startup (backend.services.detection.validate_detector); empty disables the hash check.
    default_model_sha256: str = "6c1d003cbcd03988c01b46af917e2273b3282f445e710632e72dfad7c221f276"
    default_conf: float = 0.55
    default_iou: float = 0.5
    default_tile_size: int = 1280
    default_overlap: int = 256
    default_extra_tile_sizes: list[int] = []
    default_extra_conf: list[float] = []
    default_contain: float | None = 0.8

    # Legacy detector (until 2026-11): conf 0.6, 1280 tiles only, no fragment suppression. Detection
    # jobs created before the profile was recorded used it, and the exp03 contour pipeline keeps it
    # for its own detector run (measurement of photos that have no detection job).
    legacy_model: Path = Path("models/yolo11n_tiled_best.pt")

    # Experimental contour workflow; production detector/model defaults above stay intact.
    exp03_checkpoint: Path = Path(
        "outputs/ruler_v7_reviewed367_ft_exp03_no_os/evaluation/fixed_candidate/best.pt"
    )

    # Image handling
    thumbnail_max_side: int = 2048

    # Device
    default_device: str = "0"

    model_config = {"env_prefix": "COLLEMBOLA_"}

    def ensure_dirs(self):
        """Create necessary directories."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
