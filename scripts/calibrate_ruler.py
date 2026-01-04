#!/usr/bin/env python3
"""
Semi-automated ruler calibration for microscope images.

This script helps detect the ruler in microscope images and calculate
the micrometers-per-pixel calibration factor.

Usage:
    python scripts/calibrate_ruler.py --image data/slike/K1_Fe2O3001_(1).jpg --ruler-mm 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json

def find_ruler_interactive(image_path: Path, expected_mm: float = 10.0):
    """
    Interactive ruler detection.
    
    For now, this is a placeholder that uses a manual measurement approach.
    Future versions can implement automatic ruler detection using CV techniques.
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    print(f"\nTo calibrate, you need to manually measure the ruler in pixels.")
    print(f"Expected ruler length: {expected_mm} mm")
    print(f"\nRecommendation:")
    print(f"  1. Open the image in an image viewer (e.g., ImageJ, GIMP)")
    print(f"  2. Measure the ruler length in pixels")
    print(f"  3. Enter the measurement below")
    
    # Manual input
    while True:
        try:
            ruler_px = float(input(f"\nEnter ruler length in pixels (e.g., 1167): "))
            if ruler_px <= 0:
                print("Error: Length must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid number")
    
    # Calculate calibration
    ruler_um = expected_mm * 1000  # convert mm to µm
    um_per_pixel = ruler_um / ruler_px
    
    print(f"\n{'='*70}")
    print(f"CALIBRATION RESULT")
    print(f"{'='*70}")
    print(f"Ruler length: {expected_mm} mm = {ruler_um} µm")
    print(f"Measured pixels: {ruler_px:.1f} px")
    print(f"Calibration: {um_per_pixel:.3f} µm/pixel")
    print(f"{'='*70}")
    
    # Save calibration
    cal_data = {
        'image': str(image_path),
        'ruler_mm': expected_mm,
        'ruler_um': ruler_um,
        'ruler_px': ruler_px,
        'um_per_pixel': um_per_pixel
    }
    
    cal_file = Path('data/calibration') / f"{image_path.stem}_calibration.json"
    cal_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_file, 'w') as f:
        json.dump(cal_data, f, indent=2)
    
    print(f"\n✓ Saved calibration to: {cal_file}")
    print(f"\nUse this value in measure_organisms.py:")
    print(f"  --um-per-pixel {um_per_pixel:.3f}")
    
    return um_per_pixel

def main():
    parser = argparse.ArgumentParser(
        description='Calibrate micrometers-per-pixel from ruler',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image with ruler')
    parser.add_argument('--ruler-mm', type=float, default=10.0,
                        help='Expected ruler length in millimeters (default: 10.0)')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    find_ruler_interactive(image_path, args.ruler_mm)

if __name__ == '__main__':
    main()
