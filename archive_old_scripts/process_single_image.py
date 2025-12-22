#!/usr/bin/env python3
"""
Master pipeline script for processing a single collembola plate image.

Workflow:
1. Automatic ruler calibration (with fallback to default)
2. YOLO detection (tiled inference)
3. Fast measurement (ellipse fitting)
4. Validation visualization (overview + samples)
5. Summary report

Usage:
    python scripts/process_single_image.py "data/slike/K1_Fe2O3001 (1).jpg" --output output/
    python scripts/process_single_image.py path/to/image.jpg  # uses default output/
    
Output (flat structure):
    output/
    ├── {image}_calibration.json
    ├── {image}_ruler_analysis.png
    ├── {image}_detections.csv
    ├── {image}_measurements.csv
    ├── {image}_metadata.json
    ├── {image}_overview.png
    └── {image}_samples.png
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def run_command(cmd, description):
    """Run a subprocess command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDERR: {result.stderr}")
        return False
    
    # Print stdout if available
    if result.stdout:
        print(result.stdout)
    
    return True


def load_calibration(output_dir, image_stem):
    """Load calibration data from JSON."""
    calib_path = output_dir / f"{image_stem}_calibration.json"
    if not calib_path.exists():
        return None
    
    with open(calib_path) as f:
        data = json.load(f)
    return data.get('um_per_pixel')


def load_measurements(output_dir, image_stem):
    """Load measurement statistics from CSV."""
    meas_path = output_dir / f"{image_stem}_measurements.csv"
    if not meas_path.exists():
        return None
    
    df = pd.read_csv(meas_path)
    return {
        'count': len(df),
        'mean_length_mm': df['length_mm'].mean(),
        'mean_width_mm': df['width_mm'].mean(),
        'mean_area_mm2': df['area_mm2'].mean(),
        'mean_volume_mm3': df['volume_mm3'].mean(),
        'mean_confidence': df['confidence_meas'].mean() if 'confidence_meas' in df.columns else df['confidence'].mean()
    }


def print_summary(image_path, output_dir, image_stem, elapsed_time):
    """Print final summary report."""
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    # Calibration info
    um_per_px = load_calibration(output_dir, image_stem)
    if um_per_px:
        print(f"\n✓ Calibration: {um_per_px:.3f} µm/px")
    else:
        print(f"\n✗ Calibration: Failed (used default 8.666 µm/px)")
    
    # Measurement stats
    stats = load_measurements(output_dir, image_stem)
    if stats:
        print(f"\n✓ Organisms detected: {stats['count']}")
        print(f"  Mean length:     {stats['mean_length_mm']:.3f} mm")
        print(f"  Mean width:      {stats['mean_width_mm']:.3f} mm")
        print(f"  Mean area:       {stats['mean_area_mm2']:.6f} mm²")
        print(f"  Mean volume:     {stats['mean_volume_mm3']:.6f} mm³")
        print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
    else:
        print(f"\n✗ Measurements: Failed")
    
    # Files generated
    print(f"\n✓ Output files (in {output_dir}):")
    files = [
        f"{image_stem}_calibration.json",
        f"{image_stem}_ruler_analysis.png",
        f"{image_stem}_detections.csv",
        f"{image_stem}_measurements.csv",
        f"{image_stem}_metadata.json",
        f"{image_stem}_overview.png",
        f"{image_stem}_samples.png"
    ]
    
    for fname in files:
        fpath = output_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  - {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  - {fname} (MISSING)")
    
    # Timing
    print(f"\n✓ Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Process a single collembola plate image through complete pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output/)')
    parser.add_argument('--conf', type=float, default=0.6,
                       help='YOLO confidence threshold (default: 0.6)')
    parser.add_argument('--default-cal', type=float, default=8.666,
                       help='Default µm/px if calibration fails (default: 8.666)')
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_stem = image_path.stem
    
    print(f"\nProcessing: {image_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Default calibration: {args.default_cal} µm/px")
    
    start_time = time.time()
    
    # Step 1: Ruler calibration
    success = run_command(
        ['python', 'scripts/calibrate_ruler_auto.py',
         '--image', str(image_path), 
         '--output', str(output_dir)],
        "Step 1/4: Automatic ruler calibration"
    )
    
    if not success:
        print(f"WARNING: Calibration failed, will use default {args.default_cal} µm/px")
    
    # Step 2: YOLO detection
    success = run_command(
        ['python', 'scripts/infer_tiled.py',
         '--image', str(image_path),
         '--output', str(output_dir),
         '--conf', str(args.conf)],
        "Step 2/4: YOLO organism detection"
    )
    
    if not success:
        print("ERROR: Detection failed, cannot continue")
        sys.exit(1)
    
    # Step 3: Fast measurement
    # Check if calibration exists, otherwise use default
    calib_path = output_dir / f"{image_stem}_calibration.json"
    if calib_path.exists():
        # Load calibration
        with open(calib_path) as f:
            calib_data = json.load(f)
        um_per_px = calib_data['um_per_pixel']
        
        measure_cmd = ['python', 'scripts/measure_organisms_fast.py',
                      '--image', str(image_path),
                      '--detections', str(output_dir / f"{image_stem}_detections.csv"),
                      '--um-per-pixel', str(um_per_px),
                      '--output', str(output_dir / f"{image_stem}_measurements.csv")]
    else:
        measure_cmd = ['python', 'scripts/measure_organisms_fast.py',
                      '--image', str(image_path),
                      '--detections', str(output_dir / f"{image_stem}_detections.csv"),
                      '--um-per-pixel', str(args.default_cal),
                      '--output', str(output_dir / f"{image_stem}_measurements.csv")]
    
    success = run_command(measure_cmd, "Step 3/4: Fast organism measurement")
    
    if not success:
        print("ERROR: Measurement failed, cannot continue")
        sys.exit(1)
    
    # Step 4: Validation visualization
    success = run_command(
        ['python', 'scripts/visualize_measurements.py',
         '--image', str(image_path),
         '--detections', str(output_dir / f"{image_stem}_detections.csv"),
         '--measurements', str(output_dir / f"{image_stem}_measurements.csv"),
         '--output', str(output_dir)],
        "Step 4/4: Validation visualization"
    )
    
    if not success:
        print("WARNING: Visualization failed")
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_summary(image_path, output_dir, image_stem, elapsed_time)


if __name__ == '__main__':
    main()
