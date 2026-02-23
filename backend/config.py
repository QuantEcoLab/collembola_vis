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

    # Model defaults
    default_model: Path = Path("models/yolo11n_tiled_best.pt")
    default_conf: float = 0.6
    default_iou: float = 0.5
    default_tile_size: int = 1280
    default_overlap: int = 256

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
