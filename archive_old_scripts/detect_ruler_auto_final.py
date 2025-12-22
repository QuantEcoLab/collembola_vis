#!/usr/bin/env python3
"""
Automatic ruler detection for Mid-Center and Mid-Right region.

Based on feedback that ruler is in the middle portion of image,
spanning from center to right horizontally.

Usage:
    python scripts/detect_ruler_auto_final.py \
        --image "data/slike/K1.jpg" \
        --ruler-mm 10 \
        --output calibration.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def extract_ruler_region(img: Image.Image) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Extract the Mid-Center to Mid-Right region where ruler is located.
    
    Returns:
        (region_array, x_offset, y_offset, width, height)
    """
    w, h = img.width, img.height
    
    # Mid-Center to Mid-Right spans:
    # X: from 1/3 width to full width (covers both mid-center and mid-right)
    # Y: from 1/3 height to 2/3 height (middle vertical section)
    
    x_start = w // 3
    x_end = w
    y_start = h // 3
    y_end = 2 * h // 3
    
    region = img.crop((x_start, y_start, x_end, y_end))
    region_np = np.array(region.convert('RGB'))
    
    return region_np, x_start, y_start, x_end - x_start, y_end - y_start


def find_ruler_precise(region: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the precise location of the ruler within the region.
    
    Returns:
        (x_start, x_end, y_start, y_end) relative to region, or None
    """
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    if debug:
        print(f"  Region size: {w} × {h} pixels")
    
    # Rulers typically have strong horizontal edges (ruler body)
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Sum edges horizontally to find Y position with most horizontal edges
    horizontal_edge_profile = edges.sum(axis=1)
    profile_smooth = gaussian_filter1d(horizontal_edge_profile, sigma=20)
    
    # Find peak (ruler location vertically)
    peaks, _ = find_peaks(profile_smooth, 
                         height=np.percentile(profile_smooth, 75),
                         distance=100)
    
    if len(peaks) == 0:
        if debug:
            print(f"  ⚠ No strong horizontal edge found")
        return None
    
    # Take the strongest peak
    best_peak_idx = np.argmax(profile_smooth[peaks])
    y_center = peaks[best_peak_idx]
    
    # Define ruler vertical extent (rulers are typically 150-250px tall)
    ruler_height = 200
    y_start = max(0, y_center - ruler_height // 2)
    y_end = min(h, y_center + ruler_height // 2)
    
    if debug:
        print(f"  Ruler Y range: {y_start} to {y_end}")
    
    # Now find horizontal extent
    # Look at the ruler region
    ruler_strip = gray[y_start:y_end, :]
    
    # Find vertical edges (ruler ends)
    edges_v = cv2.Canny(ruler_strip, 50, 150)
    vertical_profile = edges_v.sum(axis=0)
    profile_v_smooth = gaussian_filter1d(vertical_profile, sigma=30)
    
    # Find peaks (ruler boundaries)
    peaks_h, _ = find_peaks(profile_v_smooth,
                           height=np.percentile(profile_v_smooth, 70),
                           distance=500)
    
    if len(peaks_h) < 2:
        if debug:
            print(f"  ⚠ Could not find ruler horizontal boundaries")
        # Use full width as fallback
        x_start = 0
        x_end = w
    else:
        # Take first and last peak
        x_start = peaks_h[0]
        x_end = peaks_h[-1]
    
    if debug:
        print(f"  Ruler X range: {x_start} to {x_end}")
        print(f"  Ruler size: {x_end - x_start} × {y_end - y_start} pixels")
    
    return (x_start, x_end, y_start, y_end)


def measure_ruler_ticks(region: np.ndarray,
                       x_start: int,
                       x_end: int,
                       y_start: int,
                       y_end: int,
                       ruler_mm: float = 10.0,
                       debug: bool = False) -> Optional[Dict]:
    """
    Measure tick spacing in the ruler region.
    
    Returns:
        Dict with calibration info or None if measurement failed
    """
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    
    # Extract ruler region
    ruler_region = gray[y_start:y_end, x_start:x_end]
    
    if debug:
        print(f"\n  Analyzing ruler region ({ruler_region.shape[1]} × {ruler_region.shape[0]} px)...")
    
    # Average vertically to get horizontal intensity profile
    profile = ruler_region.mean(axis=0)
    
    # Smooth profile
    profile_smooth = gaussian_filter1d(profile, sigma=3)
    
    # Invert to find dark tick marks as peaks
    profile_inv = 255 - profile_smooth
    
    # Find peaks (tick marks)
    # Start with stricter parameters for major ticks
    threshold = np.percentile(profile_inv, 70)
    
    peaks, properties = find_peaks(profile_inv,
                                   height=threshold,
                                   distance=80,  # Major ticks should be at least 80px apart
                                   prominence=10)
    
    if len(peaks) < 5:
        # Try more lenient threshold
        threshold = np.percentile(profile_inv, 60)
        peaks, properties = find_peaks(profile_inv,
                                       height=threshold,
                                       distance=60,
                                       prominence=5)
    
    if len(peaks) < 5:
        if debug:
            print(f"  ⚠ Only found {len(peaks)} tick marks (need at least 5)")
        return None
    
    if debug:
        print(f"  ✓ Found {len(peaks)} tick marks")
    
    # Calculate spacing between consecutive ticks
    spacings = np.diff(peaks)
    
    if debug:
        print(f"  All spacings: {spacings}")
    
    # Filter for major ticks only (remove outliers)
    median_spacing = np.median(spacings)
    # Keep spacings within 50% of median
    valid_mask = np.abs(spacings - median_spacing) < 0.5 * median_spacing
    valid_spacings = spacings[valid_mask]
    
    if len(valid_spacings) < 3:
        if debug:
            print(f"  ⚠ Not enough valid spacings after filtering")
        # Use all spacings
        valid_spacings = spacings
    
    avg_spacing = np.mean(valid_spacings)
    std_spacing = np.std(valid_spacings)
    
    if debug:
        print(f"  Valid spacings: {valid_spacings}")
        print(f"  Average spacing: {avg_spacing:.1f} ± {std_spacing:.1f} px")
    
    # Calculate calibration
    # Assume ruler has major ticks at every 1mm for a 10mm ruler
    # So we should have ~11 ticks (0, 1, 2, ..., 10mm)
    # If we have N peaks, that's (N-1) intervals
    mm_per_tick = ruler_mm / (len(peaks) - 1)
    um_per_tick = mm_per_tick * 1000.0
    um_per_pixel = um_per_tick / avg_spacing
    
    if debug:
        print(f"  Calculation: {len(peaks)} ticks = {len(peaks)-1} intervals")
        print(f"  {mm_per_tick:.3f} mm/interval = {um_per_tick:.1f} µm/interval")
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
        'ruler_y_start': int(y_start),
        'ruler_y_end': int(y_end),
        'all_spacings': [int(s) for s in spacings],
        'valid_spacings': [int(s) for s in valid_spacings],
    }


def auto_calibrate_ruler_final(image_path: Path,
                               ruler_mm: float = 10.0,
                               debug: bool = False) -> Optional[Dict]:
    """
    Automatically detect and calibrate ruler in mid-center/mid-right region.
    
    Returns:
        Calibration dict or None if ruler not found
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    print(f"\nExtracting Mid-Center to Mid-Right region...")
    region, x_offset, y_offset, region_w, region_h = extract_ruler_region(img)
    print(f"Region: x={x_offset}-{x_offset + region_w}, y={y_offset}-{y_offset + region_h}")
    print(f"Region size: {region_w} × {region_h} pixels")
    
    print(f"\nSearching for ruler...")
    ruler_bounds = find_ruler_precise(region, debug=debug)
    
    if ruler_bounds is None:
        print("❌ Could not find ruler in region")
        return None
    
    x_start, x_end, y_start, y_end = ruler_bounds
    print(f"✓ Ruler found in region")
    print(f"  Local coords: x={x_start}-{x_end}, y={y_start}-{y_end}")
    print(f"  Global coords: x={x_offset + x_start}-{x_offset + x_end}, y={y_offset + y_start}-{y_offset + y_end}")
    
    print(f"\nMeasuring tick spacing...")
    result = measure_ruler_ticks(region, x_start, x_end, y_start, y_end, ruler_mm, debug=debug)
    
    if result is None:
        print("❌ Could not measure ruler tick spacing")
        return None
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC CALIBRATION RESULT")
    print(f"{'='*70}")
    print(f"Ruler location: Mid-Center to Mid-Right")
    print(f"Global position: x={x_offset + result['ruler_x_start']}-{x_offset + result['ruler_x_end']}")
    print(f"                 y={y_offset + result['ruler_y_start']}-{y_offset + result['ruler_y_end']}")
    print(f"Tick marks detected: {result['num_ticks']}")
    print(f"Average spacing: {result['spacing_px']:.1f} ± {result['spacing_std_px']:.1f} px")
    print(f"Assumed: {result['mm_per_tick']:.3f} mm per tick = {result['um_per_tick']:.1f} µm")
    print(f"→ CALIBRATION: {result['um_per_pixel']:.3f} µm/pixel")
    print(f"{'='*70}")
    
    return {
        'image': str(image_path),
        'ruler_mm': float(ruler_mm),
        'ruler_location': 'mid_center_to_mid_right',
        'global_x_start': int(x_offset + result['ruler_x_start']),
        'global_x_end': int(x_offset + result['ruler_x_end']),
        'global_y_start': int(y_offset + result['ruler_y_start']),
        'global_y_end': int(y_offset + result['ruler_y_end']),
        'num_ticks': result['num_ticks'],
        'spacing_px': result['spacing_px'],
        'spacing_std_px': result['spacing_std_px'],
        'mm_per_tick': result['mm_per_tick'],
        'um_per_tick': result['um_per_tick'],
        'um_per_pixel': result['um_per_pixel'],
        'all_spacings': result['all_spacings'],
        'valid_spacings': result['valid_spacings'],
        'method': 'automatic_mid_center_right'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Automatic ruler detection (Mid-Center to Mid-Right)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image with ruler')
    parser.add_argument('--ruler-mm', type=float, default=10.0,
                        help='Ruler length in millimeters (default: 10.0)')
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
    result = auto_calibrate_ruler_final(
        image_path=image_path,
        ruler_mm=args.ruler_mm,
        debug=args.debug
    )
    
    if result is None:
        print("\n❌ Automatic calibration failed")
        print("Try manual calibration (calibrate_ruler.py)")
        sys.exit(1)
    
    # Save calibration
    if args.output is None:
        output_dir = Path('data/calibration')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_calibration.json"
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
