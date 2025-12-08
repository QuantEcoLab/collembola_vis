#!/usr/bin/env python3
"""Segment collembolas using Segment Anything (SAM) guided by existing CSV annotations.

This script replaces the classical heuristic segmentation with SAM box prompts.
It loads the large source image once (image embedding) and sends each annotated
bounding box as a prompt. For each prompt it selects the best mask (largest
area among returned candidates) and computes basic measurements (centroid,
length, width, area, approximate volume) using the ellipsoid model from
`volumen.compute_collembola_volume`.

Expected annotation CSV: `data/collembolas_table.csv` with columns at least:
  id_collembole, x, y, w, h

Usage example:
  python sam_detect.py "data/slike/K1_Fe2O3001 (1).jpg" \
      --sam-checkpoint checkpoints/sam_vit_h.pth \
      --sam-model-type vit_h \
      --output out/K1_sam.csv --json out/K1_sam.json \
      --save-masks-dir out/masks --save-overlay out/K1_sam_overlay.png

Install dependencies (inside conda env):
  pip install torch torchvision segment-anything scikit-image pandas numpy tqdm

Model checkpoints are large; store them under `checkpoints/` (ignored by git).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from skimage import io, measure, color
from skimage.util import img_as_ubyte
from tqdm import tqdm

try:  # segment-anything import
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
except ImportError as e:  # pragma: no cover - informs user
    raise SystemExit(
        f"Failed to import segment_anything ({e}). Ensure torch + torchvision are installed.\n"
        "Install (CPU): pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install segment-anything\n"
        "Install (CUDA 12.1): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && pip install segment-anything"
    )

from volumen import compute_collembola_volume


@dataclass
class SamMeasurement:
    label: int
    centroid_x_um: float
    centroid_y_um: float
    length_um: float
    width_um: float
    length_mm: float
    volume_um3: float
    volume_mm3: float
    area_um2: float
    mask_path: str
    score: float
    iou_box: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM-based collembola segmentation.")
    p.add_argument("image", type=Path, help="Path to source image.")
    p.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/collembolas_table.csv"),
        help="CSV with bounding box annotations (default: data/collembolas_table.csv)",
    )
    p.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="Path to SAM .pth checkpoint.",
    )
    p.add_argument(
        "--sam-model-type",
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_h",
        help="SAM model variant (default: vit_h).",
    )
    p.add_argument(
        "--um-per-pixel",
        type=float,
        default=8.57,
        help="Microns per pixel calibration (default: 8.57).",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum mask area in pixels to keep (default: 100).",
    )
    p.add_argument(
        "--max-masks",
        type=int,
        default=0,
        help="Optional limit on number of prompts processed (0 = all).",
    )
    p.add_argument(
        "--box-expand",
        type=float,
        default=1.0,
        help="Multiply width/height of annotation boxes (default: 1.0).",
    )
    p.add_argument(
        "--downscale-max-side",
        type=int,
        default=0,
        help="If >0, downscale image so max side <= value before SAM embedding.",
    )
    p.add_argument(
        "--report-iou",
        action="store_true",
        help="Compute IoU of each mask with its expanded annotation box and include in outputs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="CSV file for per-collembola measurements.",
    )
    p.add_argument("--json", type=Path, help="Summary JSON output path.")
    p.add_argument(
        "--save-masks-dir",
        type=Path,
        help="Directory to save individual binary masks (PNG).",
    )
    p.add_argument(
        "--save-overlay",
        type=Path,
        help="Path to save colored overlay PNG (all masks).",
    )
    p.add_argument(
        "--image-id-column",
        type=str,
        default="id_collembole",
        help="Column used to match image rows (default: id_collembole).",
    )
    return p.parse_args(argv)


def load_annotations(
    csv_path: Path, image_path: Path, image_id_column: str
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    stem = image_path.stem
    # Partial match like existing mk_dataset approach
    subset = df[df[image_id_column].astype(str).str.contains(stem, na=False)].copy()
    required = {"x", "y", "w", "h"}
    if not required.issubset(subset.columns):
        raise SystemExit(f"Annotation CSV missing required columns: {required}")
    if subset.empty:
        print("Warning: No annotation rows matched this image.")
    return subset


def load_sam(checkpoint: Path, model_type: str):
    if not checkpoint.exists():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    predictor = SamPredictor(sam)
    return predictor


def boxes_from_annotations(df: pd.DataFrame, expand: float) -> List[List[int]]:
    boxes: List[List[int]] = []
    for _, row in df.iterrows():
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
        cx = x + w / 2.0
        cy = y + h / 2.0
        w2 = w * expand
        h2 = h * expand
        x1 = int(round(cx - w2 / 2.0))
        y1 = int(round(cy - h2 / 2.0))
        x2 = int(round(cx + w2 / 2.0))
        y2 = int(round(cy + h2 / 2.0))
        boxes.append([x1, y1, x2, y2])
    return boxes


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mask is boolean; convert to 0/255 uint8
    io.imsave(str(path), (mask.astype(np.uint8) * 255))


def measure_mask(
    mask: np.ndarray,
    um_per_pixel: float,
    label: int,
    mask_path: Path,
    score: float,
    iou_box: float,
) -> SamMeasurement:
    # Regionprops expects labeled image
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        # Fallback empty measurement
        return SamMeasurement(
            label=label,
            centroid_x_um=0.0,
            centroid_y_um=0.0,
            length_um=0.0,
            width_um=0.0,
            length_mm=0.0,
            volume_um3=0.0,
            volume_mm3=0.0,
            area_um2=0.0,
            mask_path=str(mask_path),
            score=score,
            iou_box=iou_box,
        )
    # If multiple regions due to fragmentation, merge by choosing largest region's bbox
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    height_px = maxr - minr
    width_px = maxc - minc
    length_px = (
        region.major_axis_length
        if region.major_axis_length > 0
        else max(height_px, width_px)
    )
    length_um = length_px * um_per_pixel
    width_um = width_px * um_per_pixel
    centroid_row, centroid_col = region.centroid
    centroid_x_um = centroid_col * um_per_pixel
    centroid_y_um = centroid_row * um_per_pixel
    area_um2 = region.area * (um_per_pixel**2)
    volume_um3 = compute_collembola_volume(
        width_px, height_px, um_per_pixel, model="ellipsoid"
    )
    return SamMeasurement(
        label=label,
        centroid_x_um=float(centroid_x_um),
        centroid_y_um=float(centroid_y_um),
        length_um=float(length_um),
        width_um=float(width_um),
        length_mm=float(length_um / 1000.0),
        volume_um3=float(volume_um3),
        volume_mm3=float(volume_um3 / 1_000_000_000.0),
        area_um2=float(area_um2),
        mask_path=str(mask_path),
        score=score,
        iou_box=iou_box,
    )


def write_csv(path: Path, measurements: List[SamMeasurement]) -> None:
    fieldnames = list(SamMeasurement.__annotations__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for m in measurements:
            w.writerow(asdict(m))


def write_json(
    path: Path, image: Path, um_per_pixel: float, measurements: List[SamMeasurement]
) -> None:
    payload = {
        "image": str(image),
        "um_per_pixel": um_per_pixel,
        "count": len(measurements),
        "total_length_um": sum(m.length_um for m in measurements),
        "total_volume_um3": sum(m.volume_um3 for m in measurements),
        "measurements": [asdict(m) for m in measurements],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_overlay(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    # Compose colored overlay: assign random colors
    rng = np.random.default_rng(42)
    overlay = image.copy()
    if overlay.dtype != np.float32:
        overlay = overlay.astype(np.float32)
    if overlay.max() > 1.0:
        overlay /= 255.0
    colors = rng.uniform(0.2, 1.0, size=(len(masks), 3))
    for color_vec, mask in zip(colors, masks):
        mask3 = np.stack([mask] * 3, axis=-1)
        overlay[mask3] = overlay[mask3] * 0.4 + color_vec * 0.6
    return img_as_ubyte(np.clip(overlay, 0, 1))


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    image = io.imread(str(args.image))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    ann_df = load_annotations(args.annotations, args.image, args.image_id_column)
    boxes_xyxy = boxes_from_annotations(ann_df, args.box_expand)
    if args.max_masks > 0:
        boxes_xyxy = boxes_xyxy[: args.max_masks]

    # Downscale image if requested
    scale = 1.0
    if args.downscale_max_side > 0:
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > args.downscale_max_side:
            scale = args.downscale_max_side / float(max_side)
            # Use simple bilinear resize via skimage.transform.resize (lazy import)
            from skimage.transform import resize

            image = resize(
                image,
                (int(round(h * scale)), int(round(w * scale))),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(image.dtype)
            # Scale boxes accordingly
            boxes_xyxy = [
                [
                    int(round(b[0] * scale)),
                    int(round(b[1] * scale)),
                    int(round(b[2] * scale)),
                    int(round(b[3] * scale)),
                ]
                for b in boxes_xyxy
            ]

    predictor = load_sam(args.sam_checkpoint, args.sam_model_type)
    predictor.set_image(image)

    measurements: List[SamMeasurement] = []
    kept_masks: List[np.ndarray] = []

    for i, box in enumerate(tqdm(boxes_xyxy, desc="SAM prompts")):
        masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
        areas = [m.sum() for m in masks]
        best_idx = int(np.argmax(areas))
        best_mask = masks[best_idx].astype(bool)
        if best_mask.sum() < args.min_area:
            continue
        score = float(scores[best_idx])
        # IoU with box if requested
        if args.report_iou:
            x1, y1, x2, y2 = box
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            # mask bounding box intersection
            ys, xs = np.nonzero(best_mask)
            if xs.size == 0:
                iou = 0.0
            else:
                mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                ix1 = max(x1, mx1)
                iy1 = max(y1, my1)
                ix2 = min(x2, mx2)
                iy2 = min(y2, my2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = box_area + (best_mask.sum()) - inter
                iou = float(inter / union) if union > 0 else 0.0
        else:
            iou = -1.0
        mask_path = (
            Path(args.save_masks_dir) / f"mask_{i}.png"
            if args.save_masks_dir
            else Path(f"mask_{i}.png")
        )
        if args.save_masks_dir:
            save_mask(best_mask, mask_path)
        meas = measure_mask(
            best_mask,
            args.um_per_pixel / scale,
            label=i + 1,
            mask_path=mask_path,
            score=score,
            iou_box=iou,
        )
        measurements.append(meas)
        kept_masks.append(best_mask)

    if args.save_overlay and kept_masks:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = build_overlay(image, kept_masks)
        io.imsave(str(args.save_overlay), overlay)
        print(f"Wrote overlay to {args.save_overlay}")

    # Output measurement summary to stdout
    print(f"Detected {len(measurements)} collembolas (SAM box prompts).")
    for m in measurements[:10]:  # show first few
        extra = f" IoU={m.iou_box:.3f}" if m.iou_box >= 0 else ""
        print(
            f"#{m.label} centroid=({m.centroid_x_um:.1f}µm,{m.centroid_y_um:.1f}µm) "
            f"length={m.length_um:.1f}µm volume={m.volume_um3:.1f}µm^3 score={m.score:.3f}{extra}"
        )
    if len(measurements) > 10:
        print("... (truncated) ...")

    if args.output:
        write_csv(args.output, measurements)
        print(f"Wrote CSV to {args.output}")
    if args.json:
        write_json(args.json, args.image, args.um_per_pixel, measurements)
        print(f"Wrote JSON to {args.json}")


if __name__ == "__main__":  # pragma: no cover
    main()
