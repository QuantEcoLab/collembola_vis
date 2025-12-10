from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from tqdm.auto import tqdm
from tqdm.auto import tqdm
from tqdm.auto import tqdm
from tqdm.auto import tqdm

from .config import (
    OUTPUTS_DIR,
    CSV_DIR,
    OVERLAYS_DIR,
    CLASSIFIER_THRESHOLD,
    MIN_AREA_PX,
    MIN_MAJOR_PX,
    MIN_MINOR_PX,
    MAX_AREA_PX,
    MAX_MAJOR_PX,
    MAX_MINOR_PX,
    MIN_SOLIDITY,
    MIN_ECCENTRICITY,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
)
from .segment import generate_masks
from .classify import load_classifier, is_organism, get_val_transform
from .morphology import mask_measurements
from .visualize import draw_overlay
from .preprocess import normalize_brightness, get_brightness_stats
from typing import Iterable


def crop_from_bbox(img: Image.Image, bbox, pad: int = 5) -> Image.Image:
    x, y, w, h = map(int, bbox)
    H, W = img.height, img.width
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return img.crop((x0, y0, x1, y1))


def analyze_plate(
    image_path: Path,
    device: str = "cpu",
    threshold: float = CLASSIFIER_THRESHOLD,
    verbose: bool = False,
    auto_brightness: bool = True,
) -> dict:
    print("Starting analyze_plate", flush=True)
    if verbose:
        print(f"[analyze] Segmenting {image_path.name} on {device}…", flush=True)
    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image)

    # 0) Optional brightness normalization for dark images
    if auto_brightness:
        stats_before = get_brightness_stats(np_img)
        np_img = normalize_brightness(
            np_img, target_median=50.0, auto=True, threshold=45.0
        )
        stats_after = get_brightness_stats(np_img)
        if verbose and stats_after["median"] != stats_before["median"]:
            print(
                f"[analyze] Brightness normalized: {stats_before['median']:.1f} -> {stats_after['median']:.1f}",
                flush=True,
            )

    # 1) Segment with SAM (handles optional CPU downscale internally)
    masks: List[Dict[str, Any]] = generate_masks(np_img, device=device)
    if verbose:
        print(f"[analyze] Found {len(masks)} mask proposals. Classifying…", flush=True)

    # 2) Classifier
    model = load_classifier(device=device)
    t = get_val_transform()

    # Convert normalized numpy array back to PIL Image for cropping
    image_for_crops = Image.fromarray(np_img)

    rows: List[Dict[str, Any]] = []
    accepted = 0
    iterator = enumerate(masks)
    if verbose:
        iterator = tqdm(
            enumerate(masks),
            total=len(masks),
            desc=f"Classify {image_path.stem}",
            leave=False,
        )
    for idx, m in iterator:
        bbox = m.get("bbox") or [0, 0, 0, 0]
        crop = crop_from_bbox(image_for_crops, bbox)
        ok, p = is_organism(
            crop, model, threshold=threshold, transform=t, device=device
        )
        if not ok:
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(accepted=accepted)
            continue
        meas = mask_measurements(m["segmentation"]) if "segmentation" in m else {}
        area_px = int(meas.get("area_px", 0))
        major_px = float(meas.get("major_axis_px", 0))
        minor_px = float(meas.get("minor_axis_px", 0))
        # Min-size filters to drop tiny false positives
        if area_px < MIN_AREA_PX or major_px < MIN_MAJOR_PX or minor_px < MIN_MINOR_PX:
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(accepted=accepted)
            continue
        # Max-size filters to drop large debris
        if area_px > MAX_AREA_PX or major_px > MAX_MAJOR_PX or minor_px > MAX_MINOR_PX:
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(accepted=accepted)
            continue
        # Shape filters for precision
        solidity = float(meas.get("solidity", 0.0))
        ecc = float(meas.get("eccentricity", 0.0))
        ar = (minor_px / major_px) if major_px > 0 else 1.0
        if (
            solidity < MIN_SOLIDITY
            or ecc < MIN_ECCENTRICITY
            or ar < MIN_ASPECT_RATIO
            or ar > MAX_ASPECT_RATIO
        ):
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(accepted=accepted)
            continue
        row = dict(
            plate_name=image_path.name,
            organism_id=idx,
            bbox_x=int(bbox[0]),
            bbox_y=int(bbox[1]),
            bbox_w=int(bbox[2]),
            bbox_h=int(bbox[3]),
            area_px=area_px,
            major_axis_px=major_px,
            minor_axis_px=minor_px,
            p_collembola=float(p),
        )
        row.update(meas)
        rows.append(row)
        accepted += 1
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(accepted=accepted)

    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CSV_DIR / f"{image_path.stem}_organisms.csv"
    df.to_csv(out_csv, index=False)

    # Overlay visualization
    overlay_out = OVERLAYS_DIR / f"{image_path.stem}_overlay.png"
    try:
        draw_overlay(np.array(image), masks, rows, overlay_out)
    except Exception:
        pass

    if verbose:
        print(
            f"[analyze] Done {image_path.name}: masks={len(masks)} accepted={accepted} -> {out_csv.name}",
            flush=True,
        )

    return dict(
        plate=str(image_path),
        n_masks=len(masks),
        n_organisms=int(accepted),
        csv=str(out_csv),
        overlay=str(overlay_out),
    )


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--no-auto-brightness",
        dest="auto_brightness",
        action="store_false",
        help="Disable automatic brightness normalization",
    )
    args = p.parse_args()

    res = analyze_plate(
        args.image,
        device=args.device,
        threshold=args.threshold,
        verbose=args.verbose,
        auto_brightness=args.auto_brightness,
    )
    print(json.dumps(res, indent=2))
