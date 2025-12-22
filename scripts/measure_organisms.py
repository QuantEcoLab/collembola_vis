#!/usr/bin/env python3
"""
Measure morphological properties of detected collembola organisms.

This script takes YOLO detection CSV + original image and extracts:
- Body length (µm)
- Body width (µm)  
- Area (µm²)
- Volume (µm³) using cylinder model

Uses SAM segmentation for accurate organism contours.

Usage:
    python scripts/measure_organisms.py \\
        --image data/slike/K1_Fe2O3001_(1).jpg \\
        --detections infer_tiled_output/K1_detections.csv \\
        --output measurements/K1_measurements.csv \\
        --um-per-pixel 8.57
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.transform import rotate

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("ERROR: segment_anything not installed")
    print("Install with: pip install segment-anything")
    sys.exit(1)


def load_sam_predictor(checkpoint_path: str = "checkpoints/sam_vit_b.pth", 
                       model_type: str = "vit_b",
                       device: str = "cuda") -> SamPredictor:
    """Load SAM model for mask prediction."""
    print(f"Loading SAM model from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def segment_organism_sam(predictor: SamPredictor, 
                          image: np.ndarray,
                          bbox: List[float]) -> np.ndarray:
    """
    Segment organism using SAM with bbox prompt.
    
    Args:
        predictor: SAM predictor
        image: Full image as numpy array (H, W, 3)
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Binary mask (H, W) of the organism
    """
    predictor.set_image(image)
    
    # Convert bbox to SAM format [x1, y1, x2, y2]
    input_box = np.array(bbox)
    
    # Predict mask
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )
    
    # Return best mask
    return masks[0]


def compute_cylinder_volume(length_um: float, width_um: float) -> float:
    """
    Compute volume using cylinder model.
    
    V = π * r² * h
    where r = width/2, h = length
    
    Args:
        length_um: Length in micrometers
        width_um: Width in micrometers
    
    Returns:
        Volume in cubic micrometers (µm³)
    """
    radius = width_um / 2.0
    volume = np.pi * (radius ** 2) * length_um
    return float(volume)


def measure_organism_from_mask(mask: np.ndarray, 
                                um_per_pixel: float,
                                bbox: List[float]) -> Dict[str, Any]:
    """
    Extract morphological measurements from binary mask.
    
    Args:
        mask: Binary mask of organism (full image coordinates)
        um_per_pixel: Calibration factor
        bbox: Original detection bbox [x1, y1, x2, y2]
    
    Returns:
        Dictionary with measurements
    """
    # Get region properties
    props = regionprops(mask.astype(np.uint8))
    
    if not props:
        # Fallback to bbox if mask is empty
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        length_um = max(bbox_width, bbox_height) * um_per_pixel
        width_um = min(bbox_width, bbox_height) * um_per_pixel
        
        return {
            'centroid_x_px': (x1 + x2) / 2,
            'centroid_y_px': (y1 + y2) / 2,
            'area_px': bbox_width * bbox_height,
            'area_um2': bbox_width * bbox_height * (um_per_pixel ** 2),
            'perimeter_px': 2 * (bbox_width + bbox_height),
            'major_axis_px': max(bbox_width, bbox_height),
            'minor_axis_px': min(bbox_width, bbox_height),
            'length_um': length_um,
            'width_um': width_um,
            'volume_um3': compute_cylinder_volume(length_um, width_um),
            'eccentricity': 0.0,
            'solidity': 0.0,
            'mask_available': False
        }
    
    r = props[0]
    
    # Extract measurements
    major_axis_px = float(r.major_axis_length)
    minor_axis_px = float(r.minor_axis_length)
    
    # Convert to micrometers
    length_um = major_axis_px * um_per_pixel
    width_um = minor_axis_px * um_per_pixel
    
    # Compute volume (cylinder model)
    volume_um3 = compute_cylinder_volume(length_um, width_um)
    
    # Centroid
    centroid_y, centroid_x = r.centroid
    
    return {
        'centroid_x_px': float(centroid_x),
        'centroid_y_px': float(centroid_y),
        'area_px': int(r.area),
        'area_um2': float(r.area * (um_per_pixel ** 2)),
        'perimeter_px': float(getattr(r, 'perimeter', 0.0)),
        'major_axis_px': major_axis_px,
        'minor_axis_px': minor_axis_px,
        'length_um': length_um,
        'width_um': width_um,
        'volume_um3': volume_um3,
        'eccentricity': float(r.eccentricity),
        'solidity': float(r.solidity),
        'mask_available': True
    }


def measure_organisms(image_path: Path,
                      detections_csv: Path,
                      output_csv: Path,
                      um_per_pixel: float,
                      sam_checkpoint: str = "checkpoints/sam_vit_b.pth",
                      device: str = "cuda",
                      save_visualization: bool = True):
    """
    Main function to measure all detected organisms.
    
    Args:
        image_path: Path to original plate image
        detections_csv: Path to YOLO detections CSV
        output_csv: Path to save measurements CSV
        um_per_pixel: Calibration factor (micrometers per pixel)
        sam_checkpoint: Path to SAM checkpoint
        device: Device to run SAM on
        save_visualization: Whether to save visualization overlay
    """
    # Load image
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil.convert('RGB'))
    print(f"Image size: {img_pil.width} × {img_pil.height}")
    
    # Load detections
    print(f"\nLoading detections: {detections_csv}")
    df_det = pd.read_csv(detections_csv)
    print(f"Found {len(df_det)} detections")
    
    # Load SAM
    predictor = load_sam_predictor(sam_checkpoint, device=device)
    
    # Process each detection
    print(f"\nMeasuring organisms...")
    measurements = []
    
    for idx, row in tqdm(df_det.iterrows(), total=len(df_det), desc="Processing"):
        bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
        
        try:
            # Segment organism with SAM
            mask = segment_organism_sam(predictor, img_array, bbox)
            
            # Measure from mask
            meas = measure_organism_from_mask(mask, um_per_pixel, bbox)
            
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
            print(f"Bbox: {bbox}")
            continue
    
    # Create DataFrame
    df_meas = pd.DataFrame(measurements)
    
    # Reorder columns for readability
    cols_order = [
        'detection_id',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_width_px', 'bbox_height_px',
        'centroid_x_px', 'centroid_y_px',
        'length_um', 'width_um', 'area_um2', 'volume_um3',
        'area_px', 'perimeter_px',
        'major_axis_px', 'minor_axis_px',
        'eccentricity', 'solidity',
        'confidence', 'class',
        'mask_available'
    ]
    df_meas = df_meas[cols_order]
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_meas.to_csv(output_csv, index=False)
    print(f"\n✓ Saved measurements to: {output_csv}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"MEASUREMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Organisms measured: {len(df_meas)}")
    print(f"Calibration: {um_per_pixel:.3f} µm/pixel")
    print(f"\nLength (µm):")
    print(f"  Mean:   {df_meas['length_um'].mean():.1f}")
    print(f"  Median: {df_meas['length_um'].median():.1f}")
    print(f"  Min:    {df_meas['length_um'].min():.1f}")
    print(f"  Max:    {df_meas['length_um'].max():.1f}")
    print(f"\nWidth (µm):")
    print(f"  Mean:   {df_meas['width_um'].mean():.1f}")
    print(f"  Median: {df_meas['width_um'].median():.1f}")
    print(f"\nVolume (µm³):")
    print(f"  Mean:   {df_meas['volume_um3'].mean():.1f}")
    print(f"  Median: {df_meas['volume_um3'].median():.1f}")
    print(f"  Total:  {df_meas['volume_um3'].sum():.1f}")
    print(f"{'='*70}")
    
    # Save metadata
    metadata = {
        'image_path': str(image_path),
        'detections_csv': str(detections_csv),
        'output_csv': str(output_csv),
        'um_per_pixel': um_per_pixel,
        'num_organisms': len(df_meas),
        'mean_length_um': float(df_meas['length_um'].mean()),
        'mean_width_um': float(df_meas['width_um'].mean()),
        'total_volume_um3': float(df_meas['volume_um3'].sum()),
    }
    
    metadata_path = output_csv.parent / f"{output_csv.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to: {metadata_path}")
    
    return df_meas


def main():
    parser = argparse.ArgumentParser(
        description='Measure morphological properties of detected collembola organisms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with manual calibration
  python scripts/measure_organisms.py \\
      --image data/slike/K1_Fe2O3001_(1).jpg \\
      --detections infer_tiled_output/K1_detections.csv \\
      --um-per-pixel 8.57

  # Specify output location
  python scripts/measure_organisms.py \\
      --image plate.jpg \\
      --detections plate_detections.csv \\
      --output measurements/plate_measurements.csv \\
      --um-per-pixel 8.5

  # Use CPU instead of GPU
  python scripts/measure_organisms.py \\
      --image plate.jpg \\
      --detections detections.csv \\
      --um-per-pixel 8.5 \\
      --device cpu
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to original plate image')
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to YOLO detections CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save measurements CSV (default: auto-generated)')
    parser.add_argument('--um-per-pixel', type=float, required=True,
                        help='Calibration factor: micrometers per pixel')
    parser.add_argument('--sam-checkpoint', type=str, 
                        default='checkpoints/sam_vit_b.pth',
                        help='Path to SAM checkpoint (default: checkpoints/sam_vit_b.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run SAM on: cuda or cpu (default: cuda)')
    
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
    
    # Check SAM checkpoint
    sam_checkpoint = Path(args.sam_checkpoint)
    if not sam_checkpoint.exists():
        print(f"Error: SAM checkpoint not found: {sam_checkpoint}")
        print("\nPlease download SAM checkpoint:")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth")
        sys.exit(1)
    
    # Run measurements
    measure_organisms(
        image_path=image_path,
        detections_csv=detections_csv,
        output_csv=output_csv,
        um_per_pixel=args.um_per_pixel,
        sam_checkpoint=str(sam_checkpoint),
        device=args.device
    )


if __name__ == '__main__':
    main()
