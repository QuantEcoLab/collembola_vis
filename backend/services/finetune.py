"""Fine-tuning service — builds a mini dataset from annotations and runs YOLO fine-tune."""

import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so we can import scripts
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job


def run_finetune(job: Job, progress_callback) -> dict[str, Any]:
    """Build a tiled dataset from saved annotations and fine-tune the YOLO model.

    Expected job.params:
        image_id: str
        annotation_path: str  — path to the saved annotation JSON
        base_model: str       — path to base .pt model
        epochs: int           — default 15
        device: str           — default "0"
        tile_size: int        — default 1280
        overlap: int          — default 256
    """
    from scripts.create_tiled_dataset import map_roi_to_tile, tile_image
    from ultralytics import YOLO

    params = job.params
    image_path = Path(params["image_path"])
    annotation_path = Path(params["annotation_path"])
    base_model = params.get("base_model", str(settings.default_model))
    epochs = int(params.get("epochs", 15))
    device = params.get("device", settings.default_device)
    tile_size = int(params.get("tile_size", settings.default_tile_size))
    overlap = int(params.get("overlap", settings.default_overlap))

    progress_callback(0.02, "Loading annotations")

    annotation_data = json.loads(annotation_path.read_text())
    boxes = [
        b for b in annotation_data.get("boxes", [])
        if b.get("status") in ("accepted", "added")
    ]
    if not boxes:
        raise ValueError("No accepted/added boxes in annotation file")

    # ── Build dataset dirs ────────────────────────────────────────────────────
    dataset_dir = settings.outputs_dir / job.id / "dataset"
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    progress_callback(0.05, "Tiling image")

    tiles = tile_image(str(image_path), tile_size=tile_size, overlap=overlap)
    total_tiles = len(tiles)

    # ── Map boxes to tiles ────────────────────────────────────────────────────
    annotated_tiles: list[tuple] = []
    for tile_img, x_off, y_off, tile_id in tiles:
        yolo_labels = []
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w = x2 - x1
            h = y2 - y1
            result = map_roi_to_tile(x1, y1, w, h, x_off, y_off, tile_size)
            if result is not None:
                cx, cy, nw, nh = result
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if yolo_labels:
            annotated_tiles.append((tile_img, tile_id, yolo_labels))

    if not annotated_tiles:
        raise ValueError("No tiles contained any annotated boxes")

    # ── Train/val split (80/20) ────────────────────────────────────────────────
    random.shuffle(annotated_tiles)
    n_train = max(1, int(len(annotated_tiles) * 0.8))
    splits = {
        "train": annotated_tiles[:n_train],
        "val": annotated_tiles[n_train:] or annotated_tiles[:1],
    }

    progress_callback(0.10, f"Saving {len(annotated_tiles)} annotated tiles")

    image_stem = image_path.stem
    for split, tile_list in splits.items():
        for tile_img, tile_id, labels in tile_list:
            fname = f"{image_stem}_tile{tile_id:04d}"
            tile_img.save(str(dataset_dir / "images" / split / f"{fname}.jpg"), quality=95)
            (dataset_dir / "labels" / split / f"{fname}.txt").write_text("\n".join(labels))

    # ── Write data.yaml ────────────────────────────────────────────────────────
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['collembola']\n"
    )

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    progress_callback(0.12, f"Starting fine-tune ({epochs} epochs)")

    output_dir = settings.outputs_dir / job.id / "finetune"
    model = YOLO(base_model)

    def on_epoch_end(trainer):
        frac = 0.12 + 0.80 * (trainer.epoch + 1) / epochs
        progress_callback(min(frac, 0.92), f"Epoch {trainer.epoch + 1}/{epochs}")

    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=tile_size,
        lr0=0.001,
        freeze=10,
        single_cls=True,
        project=str(output_dir),
        name="run",
        device=device,
        exist_ok=True,
    )

    progress_callback(0.93, "Saving model weights")

    # Locate best weights
    best_weights = output_dir / "run" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = output_dir / "run" / "weights" / "last.pt"

    # Copy to models/ with versioned name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"finetune_{image_stem}_{ts}.pt"
    dest = settings.models_dir / model_name
    shutil.copy2(str(best_weights), str(dest))

    # Read mAP50 from results CSV if available
    map50 = None
    results_csv = output_dir / "run" / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            # Column name varies by YOLO version
            for col in ("metrics/mAP50(B)", "metrics/mAP_0.5", "mAP_0.5"):
                if col in last:
                    try:
                        map50 = float(last[col].strip())
                    except ValueError:
                        pass
                    break

    progress_callback(1.0, f"Done — model saved as {model_name}")

    return {
        "model_name": model_name,
        "model_path": str(dest),
        "map50": map50,
        "epochs_trained": epochs,
        "tiles_used": len(annotated_tiles),
    }
