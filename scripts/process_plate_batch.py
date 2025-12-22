#!/usr/bin/env python3
"""
Batch processing for multiple collembola plates.

This script runs the complete pipeline for multiple plates:
1. YOLO tiled inference (detection)
2. SAM segmentation + morphological measurements

Usage:
    python scripts/process_plate_batch.py \\
        --images data/slike/*.jpg \\
        --model models/yolo11n_tiled_best.pt \\
        --um-per-pixel 8.57 \\
        --output-dir outputs/batch_20251210
"""

import argparse
import sys
from pathlib import Path
from typing import List
import subprocess
import json

def process_single_plate(image_path: Path,
                          model_path: Path,
                          um_per_pixel: float,
                          output_dir: Path,
                          device: str = 'cuda'):
    """Process a single plate through the complete pipeline."""
    
    plate_name = image_path.stem
    print(f"\n{'='*80}")
    print(f"Processing plate: {plate_name}")
    print(f"{'='*80}")
    
    # Create output directories
    detections_dir = output_dir / 'detections'
    measurements_dir = output_dir / 'measurements'
    overlays_dir = output_dir / 'overlays'
    
    detections_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    detections_csv = detections_dir / f"{plate_name}_detections.csv"
    measurements_csv = measurements_dir / f"{plate_name}_measurements.csv"
    
    # Step 1: YOLO inference
    print(f"\nStep 1: Running YOLO tiled inference...")
    infer_cmd = [
        'python', 'scripts/infer_tiled.py',
        '--image', str(image_path),
        '--model', str(model_path),
        '--output', str(detections_dir),
        '--device', device
    ]
    
    result = subprocess.run(infer_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in YOLO inference:")
        print(result.stderr)
        return False
    
    # Check if detections CSV was created
    if not detections_csv.exists():
        print(f"ERROR: Detection CSV not found: {detections_csv}")
        return False
    
    # Step 2: Measure organisms
    print(f"\nStep 2: Measuring organisms with SAM...")
    measure_cmd = [
        'python', 'scripts/measure_organisms.py',
        '--image', str(image_path),
        '--detections', str(detections_csv),
        '--output', str(measurements_csv),
        '--um-per-pixel', str(um_per_pixel),
        '--device', device
    ]
    
    result = subprocess.run(measure_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in measurements:")
        print(result.stderr)
        return False
    
    print(f"\n✓ Plate {plate_name} processed successfully!")
    print(f"  Detections: {detections_csv}")
    print(f"  Measurements: {measurements_csv}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple collembola plates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all plates in directory
  python scripts/process_plate_batch.py \\
      --images "data/slike/*.jpg" \\
      --model models/yolo11n_tiled_best.pt \\
      --um-per-pixel 8.57

  # Process specific plates
  python scripts/process_plate_batch.py \\
      --images data/slike/K1*.jpg data/slike/C1*.jpg \\
      --model models/yolo11n_tiled_best.pt \\
      --um-per-pixel 8.57 \\
      --output-dir outputs/experiment_1
        """
    )
    
    parser.add_argument('--images', nargs='+', required=True,
                        help='Paths to plate images (supports wildcards)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights')
    parser.add_argument('--um-per-pixel', type=float, required=True,
                        help='Calibration factor: micrometers per pixel')
    parser.add_argument('--output-dir', type=str, default='outputs/batch',
                        help='Output directory (default: outputs/batch)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Expand image paths
    image_paths = []
    for pattern in args.images:
        matches = list(Path().glob(pattern))
        if not matches:
            # Try as literal path
            p = Path(pattern)
            if p.exists():
                image_paths.append(p)
        else:
            image_paths.extend(matches)
    
    if not image_paths:
        print("ERROR: No images found matching the provided patterns")
        sys.exit(1)
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch configuration
    config = {
        'images': [str(p) for p in image_paths],
        'model': str(model_path),
        'um_per_pixel': args.um_per_pixel,
        'device': args.device,
        'num_plates': len(image_paths)
    }
    
    config_file = output_dir / 'batch_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"{'='*80}")
    print(f"BATCH PROCESSING")
    print(f"{'='*80}")
    print(f"Plates to process: {len(image_paths)}")
    print(f"Model: {model_path}")
    print(f"Calibration: {args.um_per_pixel} µm/pixel")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    # Process each plate
    success_count = 0
    failed_plates = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path.name}")
        
        success = process_single_plate(
            image_path=image_path,
            model_path=model_path,
            um_per_pixel=args.um_per_pixel,
            output_dir=output_dir,
            device=args.device
        )
        
        if success:
            success_count += 1
        else:
            failed_plates.append(image_path.name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(image_paths)} plates")
    
    if failed_plates:
        print(f"\nFailed plates:")
        for plate in failed_plates:
            print(f"  - {plate}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
