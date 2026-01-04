#!/usr/bin/env python3
"""
Fast morphological measurements using ellipse fitting (no SAM).

This script is 50-100× faster than SAM-based measurement:
- SAM: ~1 sec/organism → 800 organisms = 13+ minutes
- Ellipse: ~0.01 sec/organism → 800 organisms = 8 seconds

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


def compute_cylinder_volume(length_mm: float, width_mm: float) -> float:
    """Compute volume using cylinder model: V = π * r² * h"""
    radius = width_mm / 2.0
    volume = np.pi * (radius ** 2) * length_mm
    return float(volume)


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
    
    # Adaptive thresholding
    block_size = min(51, max(3, crop.shape[0] // 4 | 1))  # Must be odd
    thresh = filters.threshold_local(gray, block_size=block_size, offset=0.02)
    binary = gray < thresh
    
    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    
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
    
    # Get largest region
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    
    # Get measurements
    major_axis_px = float(largest.major_axis_length)
    minor_axis_px = float(largest.minor_axis_length)
    
    # Convert to millimeters (um_per_pixel gives µm, divide by 1000 for mm)
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
        'method': 'ellipse_fit'
    }


def measure_organisms_fast(image_path: Path,
                            detections_csv: Path,
                            output_csv: Path,
                            um_per_pixel: float):
    """
    Fast measurement for all detected organisms.
    """
    # Load image
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil.convert('RGB'))
    print(f"Image size: {img_pil.width} × {img_pil.height}")
    
    # Load detections
    print(f"\nLoading detections: {detections_csv}")
    df_det = pd.read_csv(detections_csv)
    print(f"Found {len(df_det)} detections")
    
    # Process each detection
    print(f"\nMeasuring organisms (fast method)...")
    measurements = []
    
    for idx, row in tqdm(df_det.iterrows(), total=len(df_det), desc="Processing"):
        bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
        
        try:
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
        'major_axis_px', 'minor_axis_px',
        'eccentricity', 'solidity',
        'confidence', 'class',
        'method'
    ]
    df_meas = df_meas[cols_order]
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_meas.to_csv(output_csv, index=False)
    print(f"\n✓ Saved measurements to: {output_csv}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"MEASUREMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Organisms measured: {len(df_meas)}")
    print(f"Calibration: {um_per_pixel:.3f} µm/pixel ({um_per_pixel/1000:.6f} mm/pixel)")
    print(f"Method: Fast ellipse fitting")
    print(f"\nLength (mm):")
    print(f"  Mean:   {df_meas['length_mm'].mean():.3f}")
    print(f"  Median: {df_meas['length_mm'].median():.3f}")
    print(f"  Min:    {df_meas['length_mm'].min():.3f}")
    print(f"  Max:    {df_meas['length_mm'].max():.3f}")
    print(f"\nWidth (mm):")
    print(f"  Mean:   {df_meas['width_mm'].mean():.3f}")
    print(f"  Median: {df_meas['width_mm'].median():.3f}")
    print(f"\nArea (mm²):")
    print(f"  Mean:   {df_meas['area_mm2'].mean():.6f}")
    print(f"  Median: {df_meas['area_mm2'].median():.6f}")
    print(f"\nVolume (mm³):")
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
    }
    
    metadata_path = output_csv.parent / f"{output_csv.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to: {metadata_path}")
    
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
