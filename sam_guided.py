#!/usr/bin/env python3
"""Guided SAM segmentation of collembolas using pooled exemplar bounding boxes.

This script treats the annotation CSV as a library of exemplar specimens (not image-specific).
It derives prototype size statistics and uses them to propose candidate regions in a new image.
Boxes are passed as prompts to Segment Anything Model (SAM). Resulting masks are filtered
and measured (length, width, volume) using skeleton-based heuristics.

Phases:
  1. Load exemplar boxes, compute prototype clusters (width/height).
  2. Build multiscale response map on target image to find peaks.
  3. Generate candidate boxes for each peak & prototype size, apply NMS.
  4. Run SAM predictor once (image embedding) and prompt with boxes.
  5. Filter & merge masks, compute measurements.
  6. Save masks, overlay, CSV & JSON summaries.

Dependencies: torch, torchvision, segment-anything, numpy, pandas, scikit-image, tqdm.

Example:
  python sam_guided.py "data/slike/K1_Fe2O3001 (1).jpg" \
      --prototypes-csv data/collembolas_table.csv \
      --sam-checkpoint checkpoints/sam_vit_b.pth --sam-model-type vit_b \
      --k-sizes 3 --peak-threshold 0.92 --max-prompts 300 \
      --save-masks-dir out/guided_masks --save-overlay out/guided_overlay.png \
      --output out/guided_measurements.csv --json out/guided_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, filters, feature, morphology, measure, color
from skimage.util import img_as_ubyte
from skimage.transform import resize

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
except ImportError as e:
    raise SystemExit(
        f"Failed to import segment_anything ({e}). Install torch + torchvision + segment-anything."
    )

from volumen import compute_collembola_volume


@dataclass
class PrototypeStats:
    widths: List[float]
    heights: List[float]
    aspects: List[float]
    median_width: float
    median_height: float
    median_aspect: float
    clusters: List[Tuple[float, float]]  # representative (w,h)


@dataclass
class Measurement:
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Guided SAM segmentation for collembolas.")
    p.add_argument("image", type=Path, help="Target image path.")
    p.add_argument(
        "--prototypes-csv",
        type=Path,
        default=Path("data/collembolas_table.csv"),
        help="CSV of exemplar boxes.",
    )
    p.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="Path to SAM checkpoint (.pth).",
    )
    p.add_argument(
        "--sam-model-type", choices=["vit_h", "vit_l", "vit_b"], default="vit_b"
    )
    p.add_argument("--um-per-pixel", type=float, default=8.57)
    p.add_argument(
        "--k-sizes",
        type=int,
        default=3,
        help="Number of prototype size clusters (k-means).",
    )
    p.add_argument(
        "--peak-threshold",
        type=float,
        default=0.9,
        help="Quantile threshold for response peaks (0-1).",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=400,
        help="Upper limit on generated box prompts.",
    )
    p.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold for non-max suppression of boxes.",
    )
    p.add_argument(
        "--downscale-max-side",
        type=int,
        default=0,
        help="Downscale image so its max side <= value (0 = no downscale).",
    )
    p.add_argument("--save-masks-dir", type=Path)
    p.add_argument("--save-overlay", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument(
        "--area-pct-min",
        type=float,
        default=5.0,
        help="Lower area percentile cutoff (default 5).",
    )
    p.add_argument(
        "--area-pct-max",
        type=float,
        default=95.0,
        help="Upper area percentile cutoff (default 95).",
    )
    p.add_argument(
        "--aspect-min",
        type=float,
        default=0.2,
        help="Minimum aspect ratio w/h (default 0.2).",
    )
    p.add_argument(
        "--aspect-max",
        type=float,
        default=5.0,
        help="Maximum aspect ratio w/h (default 5.0).",
    )
    p.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download SAM checkpoint if missing.",
    )
    p.add_argument(
        "--allow-large-image",
        action="store_true",
        help="Disable Pillow decompression bomb warning for huge images.",
    )
    return p.parse_args(argv)


def load_exemplars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"x", "y", "w", "h"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Prototype CSV must contain columns: {required}")
    return df.dropna(subset=["w", "h"]).copy()


def compute_prototypes(df: pd.DataFrame, k: int) -> PrototypeStats:
    widths = df["w"].astype(float).tolist()
    heights = df["h"].astype(float).tolist()
    aspects = [float(w) / float(h) if h > 0 else 0.0 for w, h in zip(widths, heights)]
    arr = np.column_stack([widths, heights])
    # simple k-means (numpy) initialization: random samples
    rng = np.random.default_rng(42)
    if k > arr.shape[0]:
        k = arr.shape[0]
    centroids = arr[rng.choice(arr.shape[0], size=k, replace=False)]
    for _ in range(20):  # iterations
        dists = np.linalg.norm(arr[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for ci in range(k):
            pts = arr[labels == ci]
            if pts.size:
                centroids[ci] = pts.mean(axis=0)
    clusters = [(float(c[0]), float(c[1])) for c in centroids]
    return PrototypeStats(
        widths=widths,
        heights=heights,
        aspects=aspects,
        median_width=float(np.median(widths)),
        median_height=float(np.median(heights)),
        median_aspect=float(np.median(aspects)),
        clusters=clusters,
    )


def multiscale_response(gray: np.ndarray, sigmas: List[float]) -> np.ndarray:
    # Use normalized absolute LoG responses aggregated by max
    responses = []
    for s in sigmas:
        g = filters.gaussian(gray, sigma=s)
        lap = filters.laplace(g)
        resp = np.abs(lap)
        responses.append(resp)
    stacked = np.maximum.reduce(responses)
    stacked = (stacked - stacked.min()) / (np.ptp(stacked) + 1e-9)
    return stacked


def find_peaks(resp: np.ndarray, quantile: float) -> List[Tuple[int, int]]:
    thresh = np.quantile(resp, quantile)
    mask = resp >= thresh
    # label connected high-response regions, take maxima location
    labeled = measure.label(mask)
    peaks: List[Tuple[int, int]] = []
    for region in measure.regionprops(labeled, intensity_image=resp):
        coords = region.coords
        intensities = resp[coords[:, 0], coords[:, 1]]
        max_idx = np.argmax(intensities)
        r, c = coords[max_idx]
        peaks.append((r, c))
    return peaks


def nms_boxes(
    boxes: List[List[int]], scores: List[float], iou_thresh: float
) -> List[int]:
    # boxes: [x1,y1,x2,y2]; returns indices kept
    idxs = np.argsort(scores)[::-1]
    kept = []
    while idxs.size > 0:
        i = idxs[0]
        kept.append(i)
        if idxs.size == 1:
            break
        others = idxs[1:]
        x1 = np.maximum(boxes[i][0], np.array([boxes[j][0] for j in others]))
        y1 = np.maximum(boxes[i][1], np.array([boxes[j][1] for j in others]))
        x2 = np.minimum(boxes[i][2], np.array([boxes[j][2] for j in others]))
        y2 = np.minimum(boxes[i][3], np.array([boxes[j][3] for j in others]))
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area_o = np.array([boxes[j][2] - boxes[j][0] for j in others]) * np.array(
            [boxes[j][3] - boxes[j][1] for j in others]
        )
        union = area_i + area_o - inter
        iou = inter / (union + 1e-9)
        idxs = others[iou <= iou_thresh]
    return kept


def build_candidate_boxes(
    peaks: List[Tuple[int, int]],
    prototypes: PrototypeStats,
    image_shape: Tuple[int, int],
    max_prompts: int,
) -> Tuple[List[List[int]], List[float]]:
    H, W = image_shape
    boxes: List[List[int]] = []
    scores: List[float] = []
    for r, c in peaks:
        for pw, ph in prototypes.clusters:
            w = int(round(pw))
            h = int(round(ph))
            x1 = max(0, c - w // 2)
            y1 = max(0, r - h // 2)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)
            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)  # uniform initial score; can refine later
    # Cap number of prompts
    if len(boxes) > max_prompts:
        boxes = boxes[:max_prompts]
        scores = scores[:max_prompts]
    return boxes, scores


def build_overlay(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(123)
    overlay = image.copy().astype(np.float32)
    if overlay.max() > 1.0:
        overlay /= 255.0
    colors = rng.uniform(0.2, 1.0, size=(len(masks), 3))
    for col, m in zip(colors, masks):
        # m is 2D boolean mask; select pixels (flattened N x 3) and blend
        overlay[m] = overlay[m] * 0.4 + col * 0.6
    return img_as_ubyte(np.clip(overlay, 0, 1))


def skeleton_length(mask: np.ndarray) -> float:
    skel = morphology.skeletonize(mask)
    return float(skel.sum())  # pixel count along skeleton


def measure_mask(
    mask: np.ndarray, um_per_pixel: float, label: int, path: Path
) -> Measurement:
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return Measurement(label, 0, 0, 0, 0, 0, 0, 0, 0, str(path))
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    height_px = maxr - minr
    width_px = maxc - minc
    length_px = max(region.major_axis_length, height_px, width_px)
    length_um = length_px * um_per_pixel
    width_um = width_px * um_per_pixel
    centroid_r, centroid_c = region.centroid
    centroid_x_um = centroid_c * um_per_pixel
    centroid_y_um = centroid_r * um_per_pixel
    area_um2 = region.area * (um_per_pixel**2)
    volume_um3 = compute_collembola_volume(
        width_px, height_px, um_per_pixel, model="ellipsoid"
    )
    return Measurement(
        label=label,
        centroid_x_um=centroid_x_um,
        centroid_y_um=centroid_y_um,
        length_um=length_um,
        width_um=width_um,
        length_mm=length_um / 1000.0,
        volume_um3=volume_um3,
        volume_mm3=volume_um3 / 1_000_000_000.0,
        area_um2=area_um2,
        mask_path=str(path),
    )


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if args.allow_large_image:
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
        except Exception:
            pass
    image = io.imread(str(args.image))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    exemplars = load_exemplars(args.prototypes_csv)
    prototypes = compute_prototypes(exemplars, args.k_sizes)

    # Downscale if requested
    scale = 1.0
    H, W = image.shape[:2]
    max_side = max(H, W)
    if args.downscale_max_side > 0 and max_side > args.downscale_max_side:
        scale = args.downscale_max_side / float(max_side)
        image = resize(
            image,
            (int(round(H * scale)), int(round(W * scale))),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(image.dtype)
        H, W = image.shape[:2]

    gray = color.rgb2gray(image)

    resp = multiscale_response(gray, sigmas=[2, 4, 8, 12])
    peaks = find_peaks(resp, args.peak_threshold)

    boxes, scores = build_candidate_boxes(peaks, prototypes, (H, W), args.max_prompts)
    keep = nms_boxes(boxes, scores, args.nms_iou)
    final_boxes = [boxes[i] for i in keep]

    if args.auto_download and not args.sam_checkpoint.exists():
        args.sam_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        url_map = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
        url = url_map.get(args.sam_model_type)
        if url:
            print(f"Downloading SAM checkpoint {args.sam_model_type}...")
            urllib.request.urlretrieve(url, str(args.sam_checkpoint))
            print("Download complete.")
        else:
            raise SystemExit("Unsupported model type for auto download.")
    if not args.sam_checkpoint.exists():
        raise SystemExit(f"Checkpoint not found after attempt: {args.sam_checkpoint}")
    predictor = sam_model_registry[args.sam_model_type](
        checkpoint=str(args.sam_checkpoint)
    )
    predictor = SamPredictor(predictor)
    predictor.set_image(image)

    raw_masks: List[np.ndarray] = []
    for box in tqdm(final_boxes, desc="SAM guided prompts"):
        mask_array, scores_box, _ = predictor.predict(
            box=np.array(box), multimask_output=True
        )
        areas = [m.sum() for m in mask_array]
        best_idx = int(np.argmax(areas))
        best = mask_array[best_idx].astype(bool)
        raw_masks.append(best)

    # Compute area distribution for filtering
    areas_all = np.array([m.sum() for m in raw_masks])
    if areas_all.size > 0:
        lo = np.percentile(areas_all, args.area_pct_min)
        hi = np.percentile(areas_all, args.area_pct_max)
    else:
        lo = 0
        hi = 0

    masks: List[np.ndarray] = []
    for m in raw_masks:
        area = m.sum()
        if area < lo or area > hi:
            continue
        # aspect ratio
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        aspect = w / max(h, 1)
        if aspect < args.aspect_min or aspect > args.aspect_max:
            continue
        masks.append(m)

    # Save individual masks & measurements
    measurements: List[Measurement] = []
    if args.save_masks_dir:
        args.save_masks_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(masks, start=1):
        path = (
            Path(args.save_masks_dir) / f"guided_mask_{i}.png"
            if args.save_masks_dir
            else Path(f"guided_mask_{i}.png")
        )
        if args.save_masks_dir:
            io.imsave(str(path), (m.astype(np.uint8) * 255))
        meas = measure_mask(m, args.um_per_pixel / scale, i, path)
        measurements.append(meas)

    if args.save_overlay and masks:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = build_overlay(image, masks)
        io.imsave(str(args.save_overlay), overlay)
        print(f"Wrote overlay to {args.save_overlay}")

    if args.output:
        import csv

        fieldnames = list(Measurement.__annotations__.keys())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for m in measurements:
                w.writerow(asdict(m))
        print(f"Wrote CSV to {args.output}")

    if args.json:
        payload = {
            "image": str(args.image),
            "um_per_pixel": args.um_per_pixel,
            "scale": scale,
            "prototype_clusters": prototypes.clusters,
            "median_width": prototypes.median_width,
            "median_height": prototypes.median_height,
            "count": len(measurements),
            "total_length_um": sum(m.length_um for m in measurements),
            "total_volume_um3": sum(m.volume_um3 for m in measurements),
            "measurements": [asdict(m) for m in measurements],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON to {args.json}")

    print(f"Detected {len(measurements)} collembola candidates (guided SAM).")


if __name__ == "__main__":  # pragma: no cover
    main()
