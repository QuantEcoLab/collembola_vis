#!/usr/bin/env python3
"""Template-guided SAM segmentation for collembolas.

Uses a directory of template snippets (cropped images containing individual
collembolas) to localize specimens in a large target image via normalized
cross-correlation (NCC). High-score peaks become box + point prompts for SAM.
Selected masks are filtered by size/aspect and scored. Outputs masks, overlay,
CSV, and JSON summaries.

Example:
  python sam_templates.py "data/slike/K1_Fe2O3001 (1).jpg" \
      --template-dir data/organism_templates \
      --sam-checkpoint checkpoints/sam_vit_b.pth --sam-model-type vit_b \
      --scale-factors 0.75,1.0,1.25 --ncc-threshold 0.6 --peak-distance 40 \
      --max-prompts 300 --save-masks-dir out/template_masks \
      --save-overlay out/template_overlay.png --output out/template_measurements.csv \
      --json out/template_summary.json --auto-download --allow-large-image

Dependencies: torch, torchvision, segment-anything, numpy, pandas, scikit-image, tqdm.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from skimage import io, color, feature, transform, measure, morphology
from skimage.util import img_as_ubyte
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"Failed to import segment_anything ({e}). Install torch + torchvision + segment-anything."
    )

from volumen import compute_collembola_volume


@dataclass
class TemplateStats:
    widths: List[int]
    heights: List[int]
    aspects: List[float]
    areas: List[int]
    median_width: float
    median_height: float
    p10_area: float
    p90_area: float
    p10_aspect: float
    p90_aspect: float


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
    template_score: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Template-guided SAM segmentation.")
    p.add_argument("image", type=Path, help="Target large image path.")
    p.add_argument("--template-dir", type=Path, default=Path("data/organism_templates"))
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument(
        "--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b"
    )
    p.add_argument("--um-per-pixel", type=float, default=8.57)
    p.add_argument(
        "--scale-factors",
        type=str,
        default="1.0",
        help="Comma list of template scale factors.",
    )
    p.add_argument(
        "--ncc-threshold", type=float, default=0.6, help="Min NCC score to keep a peak."
    )
    p.add_argument(
        "--peak-distance",
        type=int,
        default=30,
        help="Minimum pixel distance between NCC peaks.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=400,
        help="Upper bound on prompts sent to SAM.",
    )
    p.add_argument(
        "--downscale-max-side",
        type=int,
        default=0,
        help="Downscale large image if max side exceeds this (0=off).",
    )
    p.add_argument(
        "--min-box-overlap",
        type=float,
        default=0.4,
        help="Min ratio of mask pixels inside prompt box.",
    )
    p.add_argument("--save-masks-dir", type=Path)
    p.add_argument("--save-overlay", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument(
        "--debug-candidates",
        action="store_true",
        help="Write candidates_debug.csv with kept/discard reasons.",
    )
    p.add_argument(
        "--debug-ncc",
        action="store_true",
        help="Save NCC response maps (PNG) for each template-scale.",
    )
    p.add_argument(
        "--merge-masks", action="store_true", help="Merge nearby masks after filtering."
    )
    p.add_argument(
        "--merge-centroid-factor",
        type=float,
        default=0.6,
        help="Centroid distance factor * median template width for merging.",
    )
    p.add_argument(
        "--merge-dilate",
        type=int,
        default=1,
        help="Binary dilation iterations before intersection test.",
    )
    p.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download SAM checkpoint if missing.",
    )
    p.add_argument(
        "--allow-large-image",
        action="store_true",
        help="Disable Pillow max image warning.",
    )
    p.add_argument(
        "--max-templates",
        type=int,
        default=0,
        help="Max templates to use (0 = use all for best recall).",
    )
    return p.parse_args(argv)


def load_templates(template_dir: Path) -> List[np.ndarray]:
    if not template_dir.exists():
        raise SystemExit(f"Template directory not found: {template_dir}")
    imgs: List[np.ndarray] = []
    # Collect template images by extension (PNG/JPG/JPEG). Removed unreachable
    # union attempt with '|', which was incorrect for iterables.
    print(f"📁 Loading templates from {template_dir}")
    all_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        all_paths.extend(sorted(template_dir.glob(ext)))

    for path in tqdm(all_paths, desc="Loading templates"):
        try:
            arr = io.imread(str(path))
            if arr.ndim == 3:
                arr = color.rgb2gray(arr)
            imgs.append(arr.astype(np.float32))
        except Exception:
            continue
    if not imgs:
        raise SystemExit("No template images found.")
    print(f"✓ Loaded {len(imgs)} templates")
    return imgs


def subsample_templates(
    templates: List[np.ndarray], max_templates: int = 50
) -> List[np.ndarray]:
    """Subsample templates to reduce computation while maintaining diversity."""
    if len(templates) <= max_templates:
        return templates

    # Sample evenly across the sorted list to maintain diversity
    step = len(templates) / max_templates
    indices = [int(i * step) for i in range(max_templates)]
    sampled = [templates[i] for i in indices]
    print(
        f"📊 Subsampled {len(sampled)} from {len(templates)} templates for performance"
    )
    return sampled


def compute_template_stats(templates: List[np.ndarray]) -> TemplateStats:
    widths = [t.shape[1] for t in templates]
    heights = [t.shape[0] for t in templates]
    aspects = [w / h if h > 0 else 0.0 for w, h in zip(widths, heights)]
    areas = [w * h for w, h in zip(widths, heights)]
    areas_arr = np.array(areas)
    aspects_arr = np.array(aspects)
    stats = TemplateStats(
        widths=widths,
        heights=heights,
        aspects=aspects,
        areas=areas,
        median_width=float(np.median(widths)),
        median_height=float(np.median(heights)),
        p10_area=float(np.percentile(areas_arr, 10)),
        p90_area=float(np.percentile(areas_arr, 90)),
        p10_aspect=float(np.percentile(aspects_arr, 10)),
        p90_aspect=float(np.percentile(aspects_arr, 90)),
    )
    print(
        f"📏 Template stats: median {stats.median_width:.0f}×{stats.median_height:.0f}px"
    )
    return stats


def downscale_image(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return image, 1.0
    H, W = image.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return image, 1.0
    scale = max_side / float(m)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    if image.ndim == 3:
        resized = transform.resize(
            image, (newH, newW), preserve_range=True, anti_aliasing=True
        ).astype(image.dtype)
    else:
        resized = transform.resize(
            image, (newH, newW), preserve_range=True, anti_aliasing=True
        ).astype(image.dtype)
    return resized, scale


def match_templates(
    gray: np.ndarray,
    templates: List[np.ndarray],
    scales: List[float],
    ncc_threshold: float,
    peak_distance: int,
    max_prompts: int,
    debug_ncc_dir: Optional[Path] = None,
) -> List[Tuple[int, int, int, int, float]]:
    candidates: List[Tuple[int, int, int, int, float]] = []
    total_iterations = len(templates) * len(scales)

    print(
        f"🔍 Starting template matching: {len(templates)} templates × {len(scales)} scales = {total_iterations} iterations"
    )
    with tqdm(total=total_iterations, desc="Template matching", unit="iter") as pbar:
        for ti, tpl in enumerate(templates):
            for sf in scales:
                th = int(round(tpl.shape[0] * sf))
                tw = int(round(tpl.shape[1] * sf))
                if th < 5 or tw < 5:
                    pbar.update(1)
                    continue
                tpl_resized = transform.resize(
                    tpl, (th, tw), preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
                tpl_norm = (tpl_resized - tpl_resized.mean()) / (
                    tpl_resized.std() + 1e-9
                )
                # NCC via match_template returns response map
                resp = feature.match_template(gray, tpl_norm, pad_input=True)
                if debug_ncc_dir is not None:
                    # Normalize resp to 0-255 for PNG
                    rmin, rmax = float(resp.min()), float(resp.max())
                    if rmax > rmin:
                        resp_norm = (resp - rmin) / (rmax - rmin)
                    else:
                        resp_norm = np.zeros_like(resp)
                    from skimage.io import imsave as _imsave

                    debug_ncc_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"ncc_tpl{ti}_sf{sf:.2f}.png"
                    _imsave(
                        str(debug_ncc_dir / fname), (resp_norm * 255).astype(np.uint8)
                    )
                # Threshold
                mask = resp >= ncc_threshold
                labeled = measure.label(mask)
                for region in measure.regionprops(labeled, intensity_image=resp):
                    coords = region.coords
                    intensities = resp[coords[:, 0], coords[:, 1]]
                    max_idx = np.argmax(intensities)
                    r, c = coords[max_idx]
                    score = float(intensities[max_idx])
                    # Box centered at (r,c)
                    y1 = max(0, r - th // 2)
                    x1 = max(0, c - tw // 2)
                    y2 = min(gray.shape[0], y1 + th)
                    x2 = min(gray.shape[1], x1 + tw)
                    candidates.append((x1, y1, x2, y2, score))
                pbar.update(1)

    print(f"✓ Found {len(candidates)} candidate regions")

    # Non-max suppression by distance and score
    if not candidates:
        return []
    # Sort by score descending
    candidates.sort(key=lambda x: x[4], reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []
    print(f"🔧 Applying non-max suppression (peak_distance={peak_distance}px)...")
    for cand in candidates:
        if len(kept) >= max_prompts:
            break
        cx = (cand[0] + cand[2]) // 2
        cy = (cand[1] + cand[3]) // 2
        too_close = False
        for kc in kept:
            kcx = (kc[0] + kc[2]) // 2
            kcy = (kc[1] + kc[3]) // 2
            if (cx - kcx) ** 2 + (cy - kcy) ** 2 < peak_distance**2:
                too_close = True
                break
        if not too_close:
            kept.append(cand)
    print(f"✓ Kept {len(kept)} candidates after NMS")
    return kept


def mask_box_overlap(mask: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    inter = float(sub.sum())
    return inter / max(float(mask.sum()), 1.0)


def measure_mask(
    mask: np.ndarray, um_per_pixel: float, label: int, path: Path, template_score: float
) -> Measurement:
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return Measurement(label, 0, 0, 0, 0, 0, 0, 0, 0, str(path), template_score)
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
        template_score=template_score,
    )


def build_overlay(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(777)
    overlay = image.copy().astype(np.float32)
    if overlay.max() > 1.0:
        overlay /= 255.0
    colors = rng.uniform(0.2, 1.0, size=(len(masks), 3))
    for col, m in zip(colors, masks):
        overlay[m] = overlay[m] * 0.35 + col * 0.65
    return img_as_ubyte(np.clip(overlay, 0, 1))


def merge_masks(
    masks: List[np.ndarray],
    stats: TemplateStats,
    centroid_factor: float,
    dilate_iters: int,
) -> List[np.ndarray]:
    if not masks:
        return masks
    from skimage.morphology import binary_dilation, disk

    centroids = []
    bboxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            centroids.append((0.0, 0.0))
            bboxes.append((0, 0, 0, 0))
            continue
        centroids.append((float(xs.mean()), float(ys.mean())))
        bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    merged = [False] * len(masks)
    out: List[np.ndarray] = []
    dist_thresh = stats.median_width * centroid_factor
    dist_sq_thresh = dist_thresh * dist_thresh
    for i in range(len(masks)):
        if merged[i]:
            continue
        base = masks[i].copy()
        for j in range(i + 1, len(masks)):
            if merged[j]:
                continue
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            if dx * dx + dy * dy > dist_sq_thresh:
                continue
            # bbox touch test
            x1a, y1a, x2a, y2a = bboxes[i]
            x1b, y1b, x2b, y2b = bboxes[j]
            touching = not (
                x2a < x1b - 1 or x2b < x1a - 1 or y2a < y1b - 1 or y2b < y1a - 1
            )
            if not touching:
                # try dilation intersection
                if dilate_iters > 0:
                    selem = disk(1)
                    a_d = base
                    for _ in range(dilate_iters):
                        a_d = binary_dilation(a_d, selem)
                    b_d = masks[j]
                    for _ in range(dilate_iters):
                        b_d = binary_dilation(b_d, selem)
                    if (a_d & b_d).sum() == 0:
                        continue
                else:
                    continue
            # merge
            base = base | masks[j]
            merged[j] = True
        merged[i] = True
        out.append(base)
    return out
    rng = np.random.default_rng(777)
    overlay = image.copy().astype(np.float32)
    if overlay.max() > 1.0:
        overlay /= 255.0
    colors = rng.uniform(0.2, 1.0, size=(len(masks), 3))
    for col, m in zip(colors, masks):
        overlay[m] = overlay[m] * 0.35 + col * 0.65
    return img_as_ubyte(np.clip(overlay, 0, 1))


def auto_download_checkpoint(checkpoint: Path, model_type: str) -> None:
    if checkpoint.exists():
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    url_map = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    url = url_map.get(model_type)
    if not url:
        raise SystemExit("Unknown model type for download.")

    print(f"📥 Downloading SAM checkpoint {model_type}...")

    # Progress bar for download
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=f"Downloading {model_type}"
    ) as t:
        urllib.request.urlretrieve(url, str(checkpoint), reporthook=t.update_to)

    print(f"✓ Download complete: {checkpoint}")


def main() -> None:
    args = parse_args()
    if args.allow_large_image:
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
        except Exception:
            pass
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    print(f"\n{'=' * 60}")
    print(f"🦠 Collembola Detection with Template-Guided SAM")
    print(f"{'=' * 60}\n")

    print(f"🖼️  Loading image: {args.image}")
    image = io.imread(str(args.image))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    orig_shape = image.shape[:2]
    print(
        f"✓ Image loaded: {orig_shape[0]}×{orig_shape[1]}px ({image.nbytes / (1024**2):.1f}MB)"
    )

    # Load templates
    templates = load_templates(args.template_dir)

    # Optional: Subsample templates if explicitly requested for speed
    # By default, use ALL templates for maximum recall
    max_templates = getattr(args, "max_templates", 0)
    if max_templates > 0 and len(templates) > max_templates:
        templates = subsample_templates(templates, max_templates=max_templates)
    elif len(templates) > 100:
        print(f"⚠️  Using all {len(templates)} templates (may be slow)")
        print(f"   Consider adding --max-templates 50 for faster processing")

    stats = compute_template_stats(templates)

    # Auto-downscale huge images
    H, W = image.shape[:2]
    max_pixels = 4096 * 4096  # 16 megapixels max
    if H * W > max_pixels:
        auto_scale = args.downscale_max_side if args.downscale_max_side > 0 else 2048
        print(
            f"⚠️  Image too large ({H}×{W} = {H * W / 1e6:.1f}MP), auto-downscaling to max {auto_scale}px"
        )
        image, scale = downscale_image(image, auto_scale)
    else:
        image, scale = downscale_image(image, args.downscale_max_side)

    if scale != 1.0:
        print(
            f"↓ Downscaled to {image.shape[0]}×{image.shape[1]}px (scale={scale:.3f})"
        )

    gray = color.rgb2gray(image)
    scales = [float(s) for s in args.scale_factors.split(",") if s.strip()]
    print(f"🔢 Using {len(scales)} scale factor(s): {scales}")

    # Discover candidate boxes via template NCC
    candidates = match_templates(
        gray,
        templates,
        scales,
        args.ncc_threshold,
        args.peak_distance,
        args.max_prompts,
        debug_ncc_dir=Path("out/ncc_maps") if args.debug_ncc else None,
    )
    debug_rows = []
    if args.debug_candidates:
        # initial candidates already filtered by NMS
        for i, (x1, y1, x2, y2, score) in enumerate(candidates, start=1):
            debug_rows.append(
                {
                    "stage": "nms",
                    "reason": "kept",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": score,
                }
            )
    if not candidates:
        print("❌ No NCC candidates found above threshold.")
        return

    print(f"\n🤖 Initializing SAM model...")
    if args.auto_download:
        auto_download_checkpoint(args.sam_checkpoint, args.sam_model_type)
    if not args.sam_checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.sam_checkpoint}")

    print(f"📦 Loading SAM model ({args.sam_model_type})...")
    sam_model = sam_model_registry[args.sam_model_type](
        checkpoint=str(args.sam_checkpoint)
    )
    predictor = SamPredictor(sam_model)

    print(f"🧠 Creating image embeddings...")
    predictor.set_image(image)
    print(f"✓ SAM ready\n")

    masks: List[np.ndarray] = []
    measurements: List[Measurement] = []
    if args.save_masks_dir:
        args.save_masks_dir.mkdir(parents=True, exist_ok=True)

    for idx, (x1, y1, x2, y2, score) in enumerate(
        tqdm(candidates, desc="SAM segmentation", unit="prompt"), start=1
    ):
        box = np.array([x1, y1, x2, y2])
        # Positive point at box center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        masks_pred, scores_pred, _ = predictor.predict(
            box=box,
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        # choose mask with largest area
        areas = [m.sum() for m in masks_pred]
        best_i = int(np.argmax(areas))
        mask = masks_pred[best_i].astype(bool)
        # Overlap filtering
        overlap_ratio = mask_box_overlap(mask, (x1, y1, x2, y2))
        if overlap_ratio < args.min_box_overlap:
            if args.debug_candidates:
                debug_rows.append(
                    {
                        "stage": "mask_filter",
                        "reason": "low_overlap",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "overlap": overlap_ratio,
                    }
                )
            continue
        # Size filtering by template percentiles
        area_px = mask.sum()
        if (
            area_px < stats.p10_area or area_px > stats.p90_area * 4
        ):  # allow larger but cap extreme outliers
            if args.debug_candidates:
                debug_rows.append(
                    {
                        "stage": "mask_filter",
                        "reason": "size_outlier",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "area_px": int(area_px),
                    }
                )
            continue
        # Aspect filtering
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        aspect = w / max(h, 1)
        if aspect < stats.p10_aspect * 0.5 or aspect > stats.p90_aspect * 2.5:
            if args.debug_candidates:
                debug_rows.append(
                    {
                        "stage": "mask_filter",
                        "reason": "aspect_outlier",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "aspect": float(aspect),
                    }
                )
            continue
        if args.debug_candidates:
            debug_rows.append(
                {
                    "stage": "mask_filter",
                    "reason": "kept",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": score,
                }
            )
        masks.append(mask)
        mask_path = (
            Path(args.save_masks_dir) / f"template_mask_{idx}.png"
            if args.save_masks_dir
            else Path(f"template_mask_{idx}.png")
        )
        if args.save_masks_dir:
            io.imsave(str(mask_path), (mask.astype(np.uint8) * 255))
        meas = measure_mask(mask, args.um_per_pixel / scale, idx, mask_path, score)
        measurements.append(meas)

    if args.merge_masks and masks:
        print(f"\n🔗 Merging nearby masks...")
        merged_list = merge_masks(
            masks, stats, args.merge_centroid_factor, args.merge_dilate
        )
        if len(merged_list) != len(masks):
            print(f"✓ Merged {len(masks)} → {len(merged_list)} masks")
            # Recompute measurements for merged set
            masks = merged_list
            measurements = []
            for idx, m in enumerate(masks, start=1):
                mask_path = (
                    Path(args.save_masks_dir) / f"template_mask_merged_{idx}.png"
                    if args.save_masks_dir
                    else Path(f"template_mask_merged_{idx}.png")
                )
                if args.save_masks_dir:
                    io.imsave(str(mask_path), (m.astype(np.uint8) * 255))
                meas = measure_mask(
                    m, args.um_per_pixel / scale, idx, mask_path, template_score=0.0
                )
                measurements.append(meas)

    if args.save_overlay and masks:
        print(f"\n🎨 Creating overlay visualization...")
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = build_overlay(image, masks)
        io.imsave(str(args.save_overlay), overlay)
        print(f"✓ Overlay saved: {args.save_overlay}")

    if args.output:
        fieldnames = list(Measurement.__annotations__.keys())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in measurements:
                writer.writerow(asdict(m))
        print(f"✓ CSV saved: {args.output}")

    if args.json:
        payload = {
            "image": str(args.image),
            "um_per_pixel": args.um_per_pixel,
            "scale": scale,
            "template_dir": str(args.template_dir),
            "template_count": len(templates),
            "candidates": len(candidates),
            "kept_masks": len(masks),
            "median_width": stats.median_width,
            "median_height": stats.median_height,
            "total_length_um": sum(m.length_um for m in measurements),
            "total_volume_um3": sum(m.volume_um3 for m in measurements),
            "measurements": [asdict(m) for m in measurements],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"✓ JSON saved: {args.json}")

    if args.debug_candidates and debug_rows:
        import pandas as _pd

        _df = _pd.DataFrame(debug_rows)
        dbg_path = (
            Path("candidates_debug.csv")
            if not args.output
            else args.output.parent / "candidates_debug.csv"
        )
        _df.to_csv(dbg_path, index=False)
        print(f"✓ Debug CSV: {dbg_path}")

    print(f"\n{'=' * 60}")
    print(f"✅ Detection complete!")
    print(
        f"📊 Found {len(masks)} collembola(s) from {len(candidates)} candidate regions"
    )
    if measurements:
        total_length_mm = sum(m.length_mm for m in measurements)
        total_volume_mm3 = sum(m.volume_mm3 for m in measurements)
        print(f"📏 Total length: {total_length_mm:.2f} mm")
        print(f"📦 Total volume: {total_volume_mm3:.6f} mm³")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
