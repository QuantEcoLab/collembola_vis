"""Fine-tuning service — uses all heavily-corrected annotation files to fine-tune the model.

Only images where users manually added > MIN_ADDED_BOXES new boxes are included.
These represent the most valuable training signal: detections the model missed entirely.
"""

import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job
from backend.services.image import get_image_info

# Only include images where the user added this many new boxes
MIN_ADDED_BOXES = 10


def run_finetune_all(job: Job, progress_callback) -> dict[str, Any]:
    """Build a combined tiled dataset from all heavily-corrected annotations and fine-tune.

    Selection filter: annotation files where manually-added box count > MIN_ADDED_BOXES.
    Labels: accepted + added boxes (rejected boxes are excluded from training labels).

    Expected job.params:
        base_model: str  — path to base .pt model (default: settings.default_model)
        epochs: int      — default 20
        device: str      — default settings.default_device
        tile_size: int   — default 1280
        overlap: int     — default 256
        min_added: int   — override MIN_ADDED_BOXES threshold (default 10)
    """
    from scripts.create_tiled_dataset import map_roi_to_tile, tile_image
    from ultralytics import YOLO

    params = job.params
    base_model = params.get("base_model", str(settings.default_model))
    epochs = int(params.get("epochs", 20))
    device = params.get("device", settings.default_device)
    tile_size = int(params.get("tile_size", settings.default_tile_size))
    overlap = int(params.get("overlap", settings.default_overlap))
    min_added = int(params.get("min_added", MIN_ADDED_BOXES))

    progress_callback(0.01, "Scanning annotation files")

    ann_dir = settings.annotations_dir
    ann_files = list(ann_dir.glob("*.json"))
    if not ann_files:
        raise ValueError("No annotation files found in annotations directory")

    # ── Select qualifying annotation files ───────────────────────────────────
    qualified: list[tuple[Path, list[dict]]] = []  # (ann_path, boxes)
    skipped_low_added = 0
    skipped_no_image = 0

    for ann_path in ann_files:
        try:
            ann = json.loads(ann_path.read_text())
        except Exception:
            continue

        boxes = ann.get("boxes", [])
        added_count = sum(1 for b in boxes if b.get("status") == "added")

        if added_count <= min_added:
            skipped_low_added += 1
            continue

        image_id = ann_path.stem
        info = get_image_info(image_id)
        if info is None or not Path(info["path"]).exists():
            skipped_no_image += 1
            continue

        # Keep only accepted + added boxes as labels
        label_boxes = [b for b in boxes if b.get("status") in ("accepted", "added")]
        if not label_boxes:
            skipped_low_added += 1
            continue

        qualified.append((ann_path, label_boxes, info))

    if not qualified:
        raise ValueError(
            f"No qualifying images found. "
            f"Need added > {min_added} boxes. "
            f"Checked {len(ann_files)} files: "
            f"{skipped_low_added} had too few added boxes, "
            f"{skipped_no_image} had missing image files."
        )

    progress_callback(0.03, f"Found {len(qualified)} qualifying images — tiling")

    # ── Tile each image and map boxes ─────────────────────────────────────────
    all_tiles: list[tuple] = []  # (tile_img, fname_stem, label_lines)
    images_used = 0

    for idx, (ann_path, label_boxes, info) in enumerate(qualified):
        progress_callback(
            0.03 + 0.35 * (idx / len(qualified)),
            f"Tiling {idx + 1}/{len(qualified)}: {Path(info['path']).name}"
        )

        image_path = Path(info["path"])
        try:
            tiles = tile_image(str(image_path), tile_size=tile_size, overlap=overlap)
        except Exception as e:
            continue

        for tile_img, x_off, y_off, tile_id in tiles:
            yolo_labels = []
            for box in label_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                result = map_roi_to_tile(x1, y1, x2 - x1, y2 - y1, x_off, y_off, tile_size)
                if result is not None:
                    cx, cy, nw, nh = result
                    yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if yolo_labels:
                fname = f"{image_path.stem}_tile{tile_id:04d}"
                all_tiles.append((tile_img, fname, yolo_labels))

        images_used += 1

    if not all_tiles:
        raise ValueError("No annotated tiles produced from qualifying images")

    # ── Build dataset dirs ────────────────────────────────────────────────────
    dataset_dir = settings.outputs_dir / job.id / "dataset"
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 80/20 train/val split ─────────────────────────────────────────────────
    random.shuffle(all_tiles)
    n_train = max(1, int(len(all_tiles) * 0.8))
    splits = {
        "train": all_tiles[:n_train],
        "val": all_tiles[n_train:] or all_tiles[:1],
    }

    progress_callback(0.40, f"Saving {len(all_tiles)} tiles from {images_used} images")

    for split, tile_list in splits.items():
        for tile_img, fname, labels in tile_list:
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
    progress_callback(
        0.42,
        f"Fine-tuning {epochs} epochs · {len(all_tiles)} tiles · {images_used} images"
    )

    output_dir = settings.outputs_dir / job.id / "finetune"
    model = YOLO(base_model)

    def on_epoch_end(trainer):
        frac = 0.42 + 0.50 * (trainer.epoch + 1) / epochs
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

    best_weights = output_dir / "run" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = output_dir / "run" / "weights" / "last.pt"

    # Timestamped name — never overwrites the original model
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"finetune_all_{ts}.pt"
    dest = settings.models_dir / model_name
    shutil.copy2(str(best_weights), str(dest))

    # Read mAP50 from results CSV
    map50 = None
    results_csv = output_dir / "run" / "results.csv"
    if results_csv.exists():
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            for col in ("metrics/mAP50(B)", "metrics/mAP_0.5", "mAP_0.5"):
                if col in last:
                    try:
                        map50 = float(last[col].strip())
                    except ValueError:
                        pass
                    break

    progress_callback(1.0, f"Done — {model_name}")

    return {
        "model_name": model_name,
        "model_path": str(dest),
        "map50": map50,
        "epochs_trained": epochs,
        "tiles_used": len(all_tiles),
        "images_used": images_used,
        "min_added_threshold": min_added,
    }
