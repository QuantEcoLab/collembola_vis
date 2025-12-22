#!/usr/bin/env python3
"""
Improved automatic ruler detection focusing on bottom edge.

Based on analysis showing ruler is at bottom of image (between bottom_left
and bottom_right corners).

Usage:
    python scripts/detect_ruler_auto_v2.py \
        --image data/slike/K1.jpg \
        --ruler-mm 10 \
        --output calibration.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def extract_bottom_strip(img: Image.Image, strip_height: int = 300) -> np.ndarray:
    """
    Extract a horizontal strip from the bottom of the image containing the ruler.
    """
    # Get bottom strip
    bottom_strip = img.crop((0, img.height - strip_height, img.width, img.height))
    return np.array(bottom_strip.convert('RGB'))


def find_ruler_region(strip: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int]]:
    """
    Find the exact region where the ruler is located in the strip.
    
    Returns:
        (x_start, x_end) in pixels, or None if not found
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Rulers typically have high contrast at edges
    # Look for vertical edges (ruler boundaries)
    edges = cv2.Canny(gray, 50, 150)
    
    # Sum edges vertically to find horizontal position with most vertical edges
    vertical_edge_profile = edges.sum(axis=0)
    
    # Smooth the profile
    profile_smooth = gaussian_filter1d(vertical_edge_profile, sigma=20)
    
    # Find peaks (regions with many vertical edges = ruler sides)
    peaks, properties = find_peaks(profile_smooth, 
                                   height=np.percentile(profile_smooth, 80),
                                   distance=500)  # Ruler should be at least 500px wide
    
    if len(peaks) < 2:
        if debug:
            print(f"  ⚠ Only found {len(peaks)} ruler edges")
        return None
    
    # Take first two peaks as ruler boundaries
    x_start = peaks[0]
    x_end = peaks[1]
    
    if debug:
        print(f"  ✓ Ruler region: x={x_start} to x={x_end} (width={x_end-x_start}px)")
    
    return (x_start, x_end)


def measure_ruler_ticks(strip: np.ndarray, 
                        x_start: int, 
                        x_end: int,
                        ruler_mm: float = 10.0,
                        debug: bool = False) -> Optional[Dict]:
    """
    Measure tick spacing in the ruler region.
    
    Returns:
        Dict with calibration info or None if measurement failed
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Extract ruler region
    ruler_region = gray[:, x_start:x_end]
    
    # Average vertically to get horizontal intensity profile
    profile = ruler_region.mean(axis=0)
    
    # Smooth profile
    profile_smooth = gaussian_filter1d(profile, sigma=3)
    
    # Ruler tick marks are typically darker than background
    # Invert profile to find dark marks as peaks
    profile_inv = 255 - profile_smooth
    
    # Find peaks (tick marks)
    # Adjust height threshold based on profile statistics
    threshold = np.percentile(profile_inv, 60)
    
    peaks, properties = find_peaks(profile_inv,
                                   height=threshold,
                                   distance=30)  # Min distance between ticks
    
    if len(peaks) < 5:
        if debug:
            print(f"  ⚠ Only found {len(peaks)} tick marks (need at least 5)")
        return None
    
    if debug:
        print(f"  ✓ Found {len(peaks)} tick marks")
    
    # Calculate spacing between consecutive ticks
    spacings = np.diff(peaks)
    
    # Remove outliers (sometimes minor ticks are detected)
    median_spacing = np.median(spacings)
    # Keep spacings within 30% of median
    valid_spacings = spacings[np.abs(spacings - median_spacing) < 0.3 * median_spacing]
    
    if len(valid_spacings) < 3:
        if debug:
            print(f"  ⚠ Not enough valid spacings after outlier removal")
        return None
    
    avg_spacing = np.mean(valid_spacings)
    std_spacing = np.std(valid_spacings)
    
    if debug:
        print(f"  Tick spacing: {avg_spacing:.1f} ± {std_spacing:.1f} px")
        print(f"  Valid spacings: {len(valid_spacings)}/{len(spacings)}")
    
    # Calculate calibration
    # Assume major ticks are every 1mm for a 10mm ruler
    mm_per_tick = ruler_mm / (len(peaks) - 1)  # N peaks = N-1 intervals
    um_per_tick = mm_per_tick * 1000.0
    um_per_pixel = um_per_tick / avg_spacing
    
    if debug:
        print(f"  Calculated: {mm_per_tick:.2f} mm/tick = {um_per_tick:.1f} µm/tick")
        print(f"  Calibration: {um_per_pixel:.3f} µm/pixel")
    
    return {
        'num_ticks': int(len(peaks)),
        'spacing_px': float(avg_spacing),
        'spacing_std_px': float(std_spacing),
        'mm_per_tick': float(mm_per_tick),
        'um_per_tick': float(um_per_tick),
        'um_per_pixel': float(um_per_pixel),
        'ruler_x_start': int(x_start),
        'ruler_x_end': int(x_end),
    }


def auto_calibrate_ruler_v2(image_path: Path,
                             ruler_mm: float = 10.0,
                             strip_height: int = 300,
                             debug: bool = False) -> Optional[Dict]:
    """
    Automatically detect and calibrate ruler (V2 - bottom-focused).
    
    Returns:
        Calibration dict or None if ruler not found
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    print(f"\nExtracting bottom strip (height={strip_height}px)...")
    strip = extract_bottom_strip(img, strip_height)
    
    print(f"Searching for ruler boundaries...")
    ruler_region = find_ruler_region(strip, debug=debug)
    
    if ruler_region is None:
        print("❌ Could not find ruler boundaries")
        return None
    
    x_start, x_end = ruler_region
    print(f"✓ Ruler found at x={x_start} to x={x_end} (width={x_end-x_start}px)")
    
    print(f"\nMeasuring tick spacing...")
    result = measure_ruler_ticks(strip, x_start, x_end, ruler_mm, debug=debug)
    
    if result is None:
        print("❌ Could not measure ruler tick spacing")
        return None
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC CALIBRATION RESULT (V2)")
    print(f"{'='*70}")
    print(f"Ruler location: Bottom edge")
    print(f"Ruler position: x={result['ruler_x_start']} to x={result['ruler_x_end']}")
    print(f"Tick marks detected: {result['num_ticks']}")
    print(f"Average spacing: {result['spacing_px']:.1f} ± {result['spacing_std_px']:.1f} px")
    print(f"Assumed: {result['mm_per_tick']:.2f} mm per tick = {result['um_per_tick']:.1f} µm")
    print(f"→ Calibration: {result['um_per_pixel']:.3f} µm/pixel")
    print(f"{'='*70}")
    
    return {
        'image': str(image_path),
        'ruler_mm': float(ruler_mm),
        'ruler_location': 'bottom_edge',
        'strip_height': int(strip_height),
        'num_ticks': int(result['num_ticks']),
        'spacing_px': float(result['spacing_px']),
        'spacing_std_px': float(result['spacing_std_px']),
        'mm_per_tick': float(result['mm_per_tick']),
        'um_per_tick': float(result['um_per_tick']),
        'um_per_pixel': float(result['um_per_pixel']),
        'ruler_x_start': int(result['ruler_x_start']),
        'ruler_x_end': int(result['ruler_x_end']),
        'method': 'automatic_v2_bottom_edge'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Automatic ruler detection and calibration (V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic detection with 10mm ruler
  python scripts/detect_ruler_auto_v2.py \\
      --image "data/slike/K1_Fe2O3001 (1).jpg" \\
      --ruler-mm 10

  # Save calibration to specific location
  python scripts/detect_ruler_auto_v2.py \\
      --image plate.jpg \\
      --ruler-mm 10 \\
      --output calibration/plate_cal.json

  # Debug mode (verbose output)
  python scripts/detect_ruler_auto_v2.py \\
      --image plate.jpg \\
      --ruler-mm 10 \\
      --debug
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image with ruler')
    parser.add_argument('--ruler-mm', type=float, default=10.0,
                        help='Ruler length in millimeters (default: 10.0)')
    parser.add_argument('--strip-height', type=int, default=300,
                        help='Height of bottom strip to analyze (default: 300)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: auto-generated)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Check dependencies
    try:
        import scipy
    except ImportError:
        print("Error: scipy not installed")
        print("Install with: pip install scipy")
        sys.exit(1)
    
    # Run automatic calibration
    result = auto_calibrate_ruler_v2(
        image_path=image_path,
        ruler_mm=args.ruler_mm,
        strip_height=args.strip_height,
        debug=args.debug
    )
    
    if result is None:
        print("\n❌ Automatic calibration failed")
        print("Try:")
        print("  1. Increasing --strip-height to capture more of the ruler")
        print("  2. Using manual calibration (calibrate_ruler.py)")
        sys.exit(1)
    
    # Save calibration
    if args.output is None:
        output_dir = Path('data/calibration')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_auto_calibration.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved calibration to: {output_path}")
    print(f"\nUse this value in measure_organisms_fast.py:")
    print(f"  --um-per-pixel {result['um_per_pixel']:.3f}")


if __name__ == '__main__':
    main()
