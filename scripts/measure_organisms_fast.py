#!/usr/bin/env python3
"""
Fast morphological measurements using ellipse fitting (no SAM).

This script is 50-100x faster than SAM-based measurement:
- SAM: ~1 sec/organism -> 800 organisms = 13+ minutes
- Ellipse: ~0.01 sec/organism -> 800 organisms = 8 seconds

Method:
1. Crop bbox from image
2. Convert to grayscale + adaptive threshold
3. Find largest contour
4. Fit ellipse to get major/minor axes
5. Calculate length, width, area, volume

Usage:
    python scripts/measure_organisms_fast.py \\
        --image data/slike/K1.jpg \\
        --detections detections.csv \\
        --um-per-pixel 8.57
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage import filters, morphology, measure
from skimage.color import rgb2gray
import cv2


TRUNK_DEBUG_LIMIT = 12
SMALL_REGION_THRESHOLD_PX = 50


def compute_cylinder_volume(length_mm: float, width_mm: float) -> float:
    """Compute volume using cylinder model: V = pi * r^2 * h"""
    radius = width_mm / 2.0
    volume = np.pi * (radius ** 2) * length_mm
    return float(volume)


def _remove_small_foreground(mask: np.ndarray, threshold_px: int = SMALL_REGION_THRESHOLD_PX) -> np.ndarray:
    return morphology.remove_small_objects(mask, max_size=threshold_px - 1)


def _remove_small_holes(mask: np.ndarray, threshold_px: int = SMALL_REGION_THRESHOLD_PX) -> np.ndarray:
    return morphology.remove_small_holes(mask, max_size=threshold_px - 1)


def _region_axis_lengths(region) -> tuple[float, float]:
    try:
        return float(region.axis_major_length), float(region.axis_minor_length)
    except AttributeError:
        return float(region.major_axis_length), float(region.minor_axis_length)


def _region_mean_intensity(region) -> float:
    try:
        return float(region.intensity_mean)
    except AttributeError:
        return float(region.mean_intensity)


def _body_component_mask(gray: np.ndarray,
                         center_xy: tuple[float, float],
                         body_bbox: tuple[float, float, float, float] | None = None,
                         color_crop: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Select the pale organism body, not the adjacent dark background/halo."""
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0

    block_size = min(51, max(3, gray.shape[0] // 4 | 1))
    gray_thresh = filters.threshold_local(gray, block_size=block_size, offset=0.02)
    cx, cy = center_xy
    diag = max(1.0, float(np.hypot(gray.shape[1], gray.shape[0])))
    crop_area = max(1, gray.shape[0] * gray.shape[1])

    support = np.ones(gray.shape, dtype=bool)
    sx1, sy1, sx2, sy2 = 0, 0, gray.shape[1], gray.shape[0]
    if body_bbox is not None:
        bx1, by1, bx2, by2 = body_bbox
        margin = 4
        sx1 = max(0, int(np.floor(bx1)) - margin)
        sy1 = max(0, int(np.floor(by1)) - margin)
        sx2 = min(gray.shape[1], int(np.ceil(bx2)) + margin)
        sy2 = min(gray.shape[0], int(np.ceil(by2)) + margin)
        support = np.zeros(gray.shape, dtype=bool)
        support[sy1:sy2, sx1:sx2] = True
    support_area = max(1, int(support.sum()))

    best_score = -np.inf
    best_mask = np.zeros(gray.shape, dtype=bool)
    best_method = 'no_component'

    candidates: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    if color_crop is not None and color_crop.ndim == 3 and color_crop.size > 0:
        rgb = color_crop.astype(np.float32)
        if rgb.max() <= 1.0:
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        pale_score = value - (0.45 * saturation)
        pale_thresh = filters.threshold_local(pale_score, block_size=block_size, offset=-0.015)
        candidates.append(('pale_bright', pale_score > pale_thresh, pale_score, float(np.median(pale_score))))

    candidates.append(('bright', gray > gray_thresh, gray, float(np.median(gray))))

    for polarity, candidate, score_image, median_score in candidates:
        candidate &= support
        candidate = _remove_small_foreground(candidate)
        candidate = _remove_small_holes(candidate)
        candidate &= support
        labels = measure.label(candidate)
        if labels.max() == 0:
            continue

        for region in measure.regionprops(labels, intensity_image=score_image):
            if region.area < SMALL_REGION_THRESHOLD_PX:
                continue
            if region.area > crop_area * 0.75:
                continue
            if region.area > support_area * 0.60:
                continue

            ry, rx = region.centroid
            dist = float(np.hypot(rx - cx, ry - cy)) / diag
            area_score = min(float(region.area) / crop_area * 12.0, 1.5)
            mean_score = _region_mean_intensity(region)
            contrast = mean_score - median_score
            if contrast <= 0:
                continue
            minr, minc, maxr, maxc = region.bbox
            touches_edge = minr <= 1 or minc <= 1 or maxr >= gray.shape[0] - 1 or maxc >= gray.shape[1] - 1
            touches_support_edge = minr <= sy1 + 1 or minc <= sx1 + 1 or maxr >= sy2 - 1 or maxc >= sx2 - 1
            edge_penalty = 2.5 if touches_edge else 0.0
            support_edge_penalty = 0.8 if touches_support_edge else 0.0
            polarity_bonus = 0.55 if polarity == 'pale_bright' else 0.0
            score = (contrast * 5.0) + area_score + polarity_bonus - (dist * 2.5) - edge_penalty - support_edge_penalty

            if score > best_score:
                best_score = score
                best_mask = (labels == region.label) & support
                best_method = f'{polarity}_center_component'

    return best_mask, best_method


def extract_trunk_mask(image: np.ndarray, bbox: list) -> Dict[str, Any]:
    """Extract the conservative trunk mask used by the original fast method.

    This intentionally targets the main body only. It does not try to include legs
    or antennae, which proved unreliable with classical image processing on these
    microscope images.
    """
    x1, y1, x2, y2 = [int(x) for x in bbox]
    pad = 10
    h, w = image.shape[:2]
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(w, x2 + pad)
    crop_y2 = min(h, y2 + pad)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return {
            'mask': np.zeros((0, 0), dtype=bool),
            'crop_bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
            'contour': [],
            'area_px': 0,
            'perimeter_px': 0.0,
            'available': False,
            'method': 'empty_crop',
        }

    gray = rgb2gray(crop) if crop.ndim == 3 else crop
    center_xy = (((x1 + x2) / 2) - crop_x1, ((y1 + y2) / 2) - crop_y1)
    body_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
    trunk_mask, component_method = _body_component_mask(gray, center_xy, body_bbox, crop)
    if not trunk_mask.any():
        return {
            'mask': np.zeros(gray.shape, dtype=bool),
            'crop_bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
            'contour': [],
            'area_px': 0,
            'perimeter_px': 0.0,
            'available': False,
            'method': 'no_trunk_component',
        }

    # Smooth small edge notches while keeping the main trunk conservative.
    trunk_mask = morphology.closing(trunk_mask, morphology.disk(1))
    trunk_mask = morphology.remove_small_holes(trunk_mask, max_size=80)

    # The display contour should follow the trunk mask itself. A convex hull makes
    # crescent-shaped organisms look like large triangles, so use only mild closing
    # and dilation for a readable outline.
    contour_mask = morphology.closing(trunk_mask, morphology.disk(2))
    contour_mask = morphology.dilation(contour_mask, morphology.disk(2))

    mask_u8 = contour_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contour_points = []
        perimeter = 0.0
    else:
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        contour_points = [[int(x + crop_x1), int(y + crop_y1)] for x, y in polygon]
        perimeter = float(cv2.arcLength(contour, True))

    return {
        'mask': trunk_mask,
        'crop_bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
        'contour': contour_points,
        'area_px': int(trunk_mask.sum()),
        'perimeter_px': perimeter,
        'available': len(contour_points) > 0,
        'method': f'trunk_fast_{component_method}_contour',
    }


def _draw_trunk_overlay(image: np.ndarray, trunk_results: list[dict[str, Any]], output_path: Path) -> None:
    """Save an overlay showing conservative trunk contours."""
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for result in trunk_results:
        contour = np.array(result.get('contour', []), dtype=np.int32)
        if contour.size == 0:
            continue
        contour = contour.reshape(-1, 1, 2)
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
        x, y = contour.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(overlay, str(result['detection_id']), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _save_trunk_debug_panels(image: np.ndarray,
                             trunk_results: list[dict[str, Any]],
                             output_path: Path,
                             limit: int = TRUNK_DEBUG_LIMIT) -> None:
    """Save before/after panels for trunk contour QC."""
    panels = []
    for result in trunk_results[:limit]:
        crop_x1, crop_y1, crop_x2, crop_y2 = result['crop_bbox']
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        mask = result['mask']
        if crop.size == 0 or mask.size == 0:
            continue

        after = crop.copy()
        tint = np.zeros_like(after)
        tint[..., 1] = 255
        tint[..., 2] = 255
        after = np.where(mask[..., None], (0.65 * after + 0.35 * tint).astype(np.uint8), after)

        contour = np.array(result.get('contour', []), dtype=np.int32)
        if contour.size > 0:
            contour[:, 0] -= crop_x1
            contour[:, 1] -= crop_y1
            cv2.drawContours(after, [contour.reshape(-1, 1, 2)], -1, (255, 255, 0), 2, cv2.LINE_AA)

        target_h = 180
        scale = target_h / max(1, crop.shape[0])
        target_w = max(1, int(crop.shape[1] * scale))
        before = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, (target_w, target_h), interpolation=cv2.INTER_AREA)
        combined = np.concatenate([before, after], axis=1)
        cv2.putText(combined, f"id {result['detection_id']}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(combined)

    if not panels:
        return

    max_w = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[1] < max_w:
            panel = cv2.copyMakeBorder(panel, 0, 0, 0, max_w - panel.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245))
        padded.append(panel)
    contact_sheet = np.concatenate(padded, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def measure_organism_fast(image: np.ndarray,
                          bbox: list,
                          um_per_pixel: float) -> Dict[str, Any]:
    """
    Fast measurement using ellipse fitting.
    
    Args:
        image: Full image as numpy array (H, W, 3)
        bbox: [x1, y1, x2, y2]
        um_per_pixel: Calibration factor
    
    Returns:
        Dictionary with measurements
    """
    x1, y1, x2, y2 = [int(x) for x in bbox]
    
    # Add padding to bbox
    pad = 10
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    # Crop region
    crop = image[y1:y2, x1:x2]
    
    if crop.size == 0:
        # Fallback to bbox
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        mm_per_pixel = um_per_pixel / 1000.0
        length_mm = max(bbox_w, bbox_h) * mm_per_pixel
        width_mm = min(bbox_w, bbox_h) * mm_per_pixel
        
        return {
            'centroid_x_px': (bbox[0] + bbox[2]) / 2,
            'centroid_y_px': (bbox[1] + bbox[3]) / 2,
            'area_px': bbox_w * bbox_h,
            'area_mm2': bbox_w * bbox_h * (mm_per_pixel ** 2),
            'perimeter_px': 2 * (bbox_w + bbox_h),
            'major_axis_px': max(bbox_w, bbox_h),
            'minor_axis_px': min(bbox_w, bbox_h),
            'length_mm': length_mm,
            'width_mm': width_mm,
            'volume_mm3': compute_cylinder_volume(length_mm, width_mm),
            'eccentricity': 0.0,
            'solidity': 0.0,
            'method': 'bbox_fallback'
        }
    
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = rgb2gray(crop)
    else:
        gray = crop
    
    center_xy = (((bbox[0] + bbox[2]) / 2) - x1, ((bbox[1] + bbox[3]) / 2) - y1)
    body_bbox = (bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1)
    binary, component_method = _body_component_mask(gray, center_xy, body_bbox, crop if len(crop.shape) == 3 else None)
    
    # Find largest connected component
    labeled = measure.label(binary)
    if labeled.max() == 0:
        # No objects found, use bbox
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        mm_per_pixel = um_per_pixel / 1000.0
        length_mm = max(bbox_w, bbox_h) * mm_per_pixel
        width_mm = min(bbox_w, bbox_h) * mm_per_pixel
        
        return {
            'centroid_x_px': (bbox[0] + bbox[2]) / 2,
            'centroid_y_px': (bbox[1] + bbox[3]) / 2,
            'area_px': bbox_w * bbox_h,
            'area_mm2': bbox_w * bbox_h * (mm_per_pixel ** 2),
            'perimeter_px': 2 * (bbox_w + bbox_h),
            'major_axis_px': max(bbox_w, bbox_h),
            'minor_axis_px': min(bbox_w, bbox_h),
            'length_mm': length_mm,
            'width_mm': width_mm,
            'volume_mm3': compute_cylinder_volume(length_mm, width_mm),
            'eccentricity': 0.0,
            'solidity': 0.0,
            'method': 'no_contour_fallback'
        }
    
    # Get selected body region
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    
    # Get measurements
    major_axis_px, minor_axis_px = _region_axis_lengths(largest)
    
    # Convert to millimeters (um_per_pixel gives micrometers, divide by 1000 for mm)
    mm_per_pixel = um_per_pixel / 1000.0
    length_mm = major_axis_px * mm_per_pixel
    width_mm = minor_axis_px * mm_per_pixel
    
    # Calculate volume
    volume_mm3 = compute_cylinder_volume(length_mm, width_mm)
    
    # Centroid in global coordinates
    centroid_y, centroid_x = largest.centroid
    global_centroid_x = x1 + centroid_x
    global_centroid_y = y1 + centroid_y
    
    return {
        'centroid_x_px': float(global_centroid_x),
        'centroid_y_px': float(global_centroid_y),
        'area_px': int(largest.area),
        'area_mm2': float(largest.area * (mm_per_pixel ** 2)),
        'perimeter_px': float(getattr(largest, 'perimeter', 0.0)),
        'major_axis_px': major_axis_px,
        'minor_axis_px': minor_axis_px,
        'length_mm': length_mm,
        'width_mm': width_mm,
        'volume_mm3': volume_mm3,
        'eccentricity': float(largest.eccentricity),
        'solidity': float(largest.solidity),
        'method': f'ellipse_fit_{component_method}'
    }


def measure_organisms_fast(image_path: Path,
                            detections_csv: Path,
                            output_csv: Path,
                            um_per_pixel: float,
                            progress_callback=None):
    """
    Fast measurement for all detected organisms.

    Args:
        image_path: Path to original plate image
        detections_csv: Path to YOLO detections CSV
        output_csv: Path to save measurements CSV
        um_per_pixel: Calibration factor (micrometers per pixel)
        progress_callback: Optional callable(progress: float, message: str) for progress updates
    """
    # Load image
    print(f"Loading image: {image_path}")
    if progress_callback:
        progress_callback(0.0, "Loading image...")
    Image.MAX_IMAGE_PIXELS = None
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil.convert('RGB'))
    print(f"Image size: {img_pil.width} x {img_pil.height}")

    # Load detections
    print(f"\nLoading detections: {detections_csv}")
    df_det = pd.read_csv(detections_csv)
    print(f"Found {len(df_det)} detections")

    # Process each detection
    print(f"\nMeasuring organisms (fast method)...")
    measurements = []
    trunk_results = []
    trunk_masks = {}

    total = len(df_det)
    if progress_callback:
        progress_callback(0.05, f"Measuring {total} organisms...")
    for idx, row in tqdm(df_det.iterrows(), total=total, desc="Processing"):
        bbox = [row['x1'], row['y1'], row['x2'], row['y2']]

        if progress_callback:
            progress_callback(0.05 + 0.90 * (idx / total), f"Organism {idx + 1}/{total}")

        try:
            trunk = extract_trunk_mask(img_array, bbox)
            trunk['detection_id'] = int(idx)
            trunk_results.append(trunk)
            if trunk['mask'].size > 0:
                trunk_masks[f"mask_{idx}"] = trunk['mask'].astype(np.uint8)

            meas = measure_organism_fast(img_array, bbox, um_per_pixel)
            
            # Add detection info
            meas['detection_id'] = int(idx)
            meas['bbox_x1'] = row['x1']
            meas['bbox_y1'] = row['y1']
            meas['bbox_x2'] = row['x2']
            meas['bbox_y2'] = row['y2']
            meas['bbox_width_px'] = row['width']
            meas['bbox_height_px'] = row['height']
            meas['confidence'] = row['confidence']
            meas['class'] = row['class']
            meas['trunk_mask_available'] = bool(trunk['available'])
            meas['trunk_area_px'] = trunk['area_px']
            meas['trunk_area_mm2'] = float(trunk['area_px'] * ((um_per_pixel / 1000.0) ** 2))
            meas['trunk_perimeter_px'] = trunk['perimeter_px']
            meas['trunk_contour_points'] = len(trunk['contour'])
            
            measurements.append(meas)
            
        except Exception as e:
            print(f"\nWarning: Failed to measure detection {idx}: {e}")
            continue
    
    # Create DataFrame
    df_meas = pd.DataFrame(measurements)
    
    # Reorder columns
    cols_order = [
        'detection_id',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_width_px', 'bbox_height_px',
        'centroid_x_px', 'centroid_y_px',
        'length_mm', 'width_mm', 'area_mm2', 'volume_mm3',
        'area_px', 'perimeter_px',
        'trunk_area_px', 'trunk_area_mm2', 'trunk_perimeter_px',
        'trunk_contour_points', 'trunk_mask_available',
        'major_axis_px', 'minor_axis_px',
        'eccentricity', 'solidity',
        'confidence', 'class',
        'method'
    ]
    if df_meas.empty:
        df_meas = pd.DataFrame(columns=cols_order)
    else:
        df_meas = df_meas[cols_order]

    if progress_callback:
        progress_callback(1.0, f"Done - {len(df_meas)} organisms measured")

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_meas.to_csv(output_csv, index=False)
    print(f"\nSaved measurements to: {output_csv}")

    # Save conservative trunk masks and contours. These are the recommended
    # classical-CV contours for this dataset; full-body appendages are not reliable.
    trunk_masks_path = output_csv.parent / f"{output_csv.stem}_trunk_masks.npz"
    if trunk_masks:
        np.savez_compressed(trunk_masks_path, **trunk_masks)
        print(f"Saved trunk masks to: {trunk_masks_path}")

    trunk_contours_payload = {
        'image_path': str(image_path),
        'detections_csv': str(detections_csv),
        'coordinate_system': 'global_image_pixels',
        'mask_file': str(trunk_masks_path) if trunk_masks else None,
        'organisms': [
            {
                'detection_id': r['detection_id'],
                'crop_bbox': r['crop_bbox'],
                'mask_key': f"mask_{r['detection_id']}" if r['mask'].size > 0 else None,
                'trunk_area_px': r['area_px'],
                'trunk_perimeter_px': r['perimeter_px'],
                'trunk_mask_available': r['available'],
                'trunk_contour': r['contour'],
                'method': r['method'],
            }
            for r in trunk_results
        ],
    }
    trunk_contours_path = output_csv.parent / f"{output_csv.stem}_trunk_contours.json"
    with open(trunk_contours_path, 'w') as f:
        json.dump(trunk_contours_payload, f, indent=2)
    print(f"Saved trunk contours to: {trunk_contours_path}")

    trunk_overlay_path = output_csv.parent / f"{image_path.stem}_trunk_overlay.jpg"
    _draw_trunk_overlay(img_array, trunk_results, trunk_overlay_path)
    print(f"Saved trunk overlay to: {trunk_overlay_path}")

    trunk_debug_path = output_csv.parent / f"{image_path.stem}_trunk_debug_panels.jpg"
    _save_trunk_debug_panels(img_array, trunk_results, trunk_debug_path)
    if trunk_debug_path.exists():
        print(f"Saved trunk before/after debug panels to: {trunk_debug_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"MEASUREMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Organisms measured: {len(df_meas)}")
    print(f"Calibration: {um_per_pixel:.3f} um/pixel ({um_per_pixel/1000:.6f} mm/pixel)")
    print(f"Method: Fast ellipse fitting")
    print(f"\nLength (mm):")
    print(f"  Mean:   {df_meas['length_mm'].mean():.3f}")
    print(f"  Median: {df_meas['length_mm'].median():.3f}")
    print(f"  Min:    {df_meas['length_mm'].min():.3f}")
    print(f"  Max:    {df_meas['length_mm'].max():.3f}")
    print(f"\nWidth (mm):")
    print(f"  Mean:   {df_meas['width_mm'].mean():.3f}")
    print(f"  Median: {df_meas['width_mm'].median():.3f}")
    print(f"\nArea (mm^2):")
    print(f"  Mean:   {df_meas['area_mm2'].mean():.6f}")
    print(f"  Median: {df_meas['area_mm2'].median():.6f}")
    print(f"\nVolume (mm^3):")
    print(f"  Mean:   {df_meas['volume_mm3'].mean():.6f}")
    print(f"  Median: {df_meas['volume_mm3'].median():.6f}")
    print(f"  Total:  {df_meas['volume_mm3'].sum():.6f}")
    print(f"{'='*70}")
    
    # Save metadata
    metadata = {
        'image_path': str(image_path),
        'detections_csv': str(detections_csv),
        'output_csv': str(output_csv),
        'um_per_pixel': um_per_pixel,
        'mm_per_pixel': um_per_pixel / 1000.0,
        'method': 'fast_ellipse_fitting',
        'num_organisms': len(df_meas),
        'mean_length_mm': float(df_meas['length_mm'].mean()),
        'mean_width_mm': float(df_meas['width_mm'].mean()),
        'mean_area_mm2': float(df_meas['area_mm2'].mean()),
        'total_volume_mm3': float(df_meas['volume_mm3'].sum()),
        'trunk_masks_path': str(trunk_masks_path) if trunk_masks else None,
        'trunk_contours_path': str(trunk_contours_path),
        'trunk_overlay_path': str(trunk_overlay_path),
        'trunk_debug_panels_path': str(trunk_debug_path) if trunk_debug_path.exists() else None,
    }
    
    metadata_path = output_csv.parent / f"{output_csv.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")
    
    return df_meas


def main():
    parser = argparse.ArgumentParser(
        description='Fast morphological measurements using ellipse fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to original plate image')
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to YOLO detections CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save measurements CSV (default: auto-generated)')
    parser.add_argument('--um-per-pixel', type=float, required=True,
                        help='Calibration factor: micrometers per pixel')
    
    args = parser.parse_args()
    
    # Paths
    image_path = Path(args.image)
    detections_csv = Path(args.detections)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not detections_csv.exists():
        print(f"Error: Detections CSV not found: {detections_csv}")
        sys.exit(1)
    
    # Auto-generate output path if not specified
    if args.output is None:
        output_csv = Path('measurements') / f"{image_path.stem}_measurements.csv"
    else:
        output_csv = Path(args.output)
    
    # Run measurements
    measure_organisms_fast(
        image_path=image_path,
        detections_csv=detections_csv,
        output_csv=output_csv,
        um_per_pixel=args.um_per_pixel
    )


if __name__ == '__main__':
    main()
