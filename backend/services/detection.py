"""Detection service — wraps scripts/infer_tiled.py."""

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so we can import scripts
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job
from backend.services.cuda_guard import ensure_inference_device
from backend.services.image_identity import capture_identity

PROFILE_KEYS = ("conf", "iou", "tile_size", "overlap", "extra_tile_sizes", "extra_conf", "contain")


def detection_profile(params: dict) -> dict[str, Any]:
    """Detector settings a job runs with: request values over the current settings defaults."""
    contain = params.get("contain", settings.default_contain)
    return {
        "model_path": str(params.get("model_path", settings.default_model)),
        "conf": float(params.get("conf", settings.default_conf)),
        "iou": float(params.get("iou", settings.default_iou)),
        "tile_size": int(params.get("tile_size", settings.default_tile_size)),
        "overlap": int(params.get("overlap", settings.default_overlap)),
        "extra_tile_sizes": [int(v) for v in params.get("extra_tile_sizes", settings.default_extra_tile_sizes)],
        "extra_conf": [float(v) for v in params.get("extra_conf", settings.default_extra_conf)],
        "contain": None if contain is None else float(contain),
    }


def legacy_profile() -> dict[str, Any]:
    """The detector settings used before the profile was recorded with each job."""
    return {"model_path": str(settings.legacy_model), "conf": 0.6, "iou": 0.5, "tile_size": 1280, "overlap": 256,
            "extra_tile_sizes": [], "extra_conf": [], "contain": None}


def recorded_profile(job: Job, metadata: dict | None = None) -> dict[str, Any]:
    """Profile of a completed detection job; jobs from before profiles were recorded used the legacy
    detector, with any explicitly requested values taken from their params (or metadata file)."""
    if job.result.get("detector_profile"):
        return dict(job.result["detector_profile"])
    profile = legacy_profile()
    source = {k: v for k, v in (job.params or {}).items() if k in PROFILE_KEYS or k == "model_path"}
    if not source and metadata:
        source = {"conf": metadata.get("conf_threshold"), "iou": metadata.get("iou_threshold"),
                  "tile_size": metadata.get("tile_size"), "overlap": metadata.get("overlap"),
                  "model_path": metadata.get("model_path")}
    profile.update({k: v for k, v in source.items() if v is not None})
    return profile


def same_profile(a: dict, b: dict) -> bool:
    """Profiles match when every setting and the resolved model file are equal."""
    def norm(p):
        return {**{k: p.get(k) for k in PROFILE_KEYS}, "model_path": str(Path(p["model_path"]).resolve())}
    return norm(a) == norm(b)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def validate_detector() -> Path:
    """Startup check: the standard detector weights are installed and are the expected build.

    The weights are not in git; a deploy that forgets to copy them would otherwise fail every
    standard detection (and the fine-tune default) at run time instead of at startup.
    """
    import hashlib

    path = _repo_path(settings.default_model)
    if not path.is_file():
        raise RuntimeError(
            f"Standard detector weights are missing: {path}. Install the file named by default_model "
            f"(sha256 {settings.default_model_sha256}) before starting the service; see docs/DETECTION_RECALL.md.")
    if settings.default_model_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != settings.default_model_sha256:
            raise RuntimeError(f"Standard detector {path} has sha256 {digest}, expected {settings.default_model_sha256}.")
    return path


def run_detection(job: Job, progress_callback) -> dict[str, Any]:
    """Execute tiled inference for a job.

    Expected job.params:
        image_path: str
        model_path: str (optional, default from settings)
        conf: float (optional)
        iou: float (optional)
        tile_size: int (optional)
        overlap: int (optional)
        extra_tile_sizes: list[int] (optional)
        extra_conf: list[float] (optional)
        contain: float | None (optional)
        device: str (optional)

    Missing values come from the settings defaults; the resolved values are returned as
    ``detector_profile``.
    """
    from scripts.infer_tiled import infer_tiled

    params = job.params
    image_path = Path(params["image_path"])
    identity = capture_identity(params.get("image_id", image_path.parent.name), image_path)
    expected = params.get("image_identity")
    if expected and expected != identity:
        raise ValueError("Fotografija ne odgovara identitetu zabilježenom pri slanju detekcije")
    output_dir = settings.outputs_dir / job.id
    device = params.get("device", settings.default_device)
    ensure_inference_device(device)
    profile = detection_profile(params)

    detections = infer_tiled(
        image_path=image_path,
        model_path=profile["model_path"],
        tile_size=profile["tile_size"],
        overlap=profile["overlap"],
        conf_threshold=profile["conf"],
        iou_threshold=profile["iou"],
        output_dir=str(output_dir),
        device=device,
        progress_callback=progress_callback,
        extra_tile_sizes=profile["extra_tile_sizes"],
        extra_conf_thresholds=profile["extra_conf"],
        contain_threshold=profile["contain"],
    )

    image_stem = image_path.stem
    csv_path = output_dir / f"{image_stem}_detections.csv"
    overlay_path = output_dir / f"{image_stem}_overlay.jpg"
    metadata_path = output_dir / f"{image_stem}_metadata.json"

    if capture_identity(identity["image_id"], image_path) != identity:
        raise ValueError("Fotografija je promijenjena tijekom detekcije")
    return {
        "image_id": identity["image_id"],
        "image_identity": identity,
        "num_detections": len(detections),
        "csv_path": str(csv_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
        "image_path": str(image_path),
        "image_stem": image_stem,
        "detector_profile": profile,
    }
