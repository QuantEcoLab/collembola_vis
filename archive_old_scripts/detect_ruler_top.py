#!/usr/bin/env python3
"""
Automatic ruler detection focusing on TOP of image.

Updated based on feedback that ruler is in top center of image.

Usage:
    python scripts/detect_ruler_top.py \
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
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def extract_top_strip(img: Image.Image, strip_height: int = 500) -> np.ndarray:
    """
    Extract a horizontal strip from the TOP of the image containing the ruler.
    """
    # Get top strip
    top_strip = img.crop((0, 0, img.width, strip_height))
    return np.array(top_strip.convert('RGB'))


def find_ruler_region_top(strip: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the exact region where the ruler is located in the top strip.
    Ruler is approximately in the center horizontally.
    
    Returns:
        (x_start, x_end, y_start, y_end) in pixels, or None if not found
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Since ruler is in top center, focus on center region horizontally
    # and scan vertically to find ruler
    center_x = w // 2
    search_width = w // 3  # Search ±1/6 of image width around center
    x_min = max(0, center_x - search_width // 2)
    x_max = min(w, center_x + search_width // 2)
    
    if debug:
        print(f"  Searching horizontal range: x={x_min} to x={x_max}")
    
    # Look at the center region
    center_region = gray[:, x_min:x_max]
    
    # Rulers typically have strong horizontal edges
    edges = cv2.Canny(center_region, 50, 150)
    
    # Sum edges horizontally to find vertical position with most horizontal edges
    horizontal_edge_profile = edges.sum(axis=1)
    
    # Smooth the profile
    profile_smooth = gaussian_filter1d(horizontal_edge_profile, sigma=10)
    
    # Find peaks (regions with many horizontal edges = ruler location)
    peaks, properties = find_peaks(profile_smooth, 
                                   height=np.percentile(profile_smooth, 70),
                                   distance=50)
    
    if len(peaks) < 1:
        if debug:
            print(f"  ⚠ No strong horizontal edge regions found")
        return None
    
    # Take the highest peak as ruler location
    best_peak_idx = np.argmax(profile_smooth[peaks])
    y_center = peaks[best_peak_idx]
    
    # Define ruler region (assume ruler is ~200px tall)
    ruler_height = 200
    y_start = max(0, y_center - ruler_height // 2)
    y_end = min(h, y_center + ruler_height // 2)
    
    # Now find horizontal extent of ruler
    ruler_row = gray[y_start:y_end, :]
    
    # Look for vertical edges (ruler ends)
    edges_full = cv2.Canny(ruler_row, 50, 150)
    vertical_edge_profile = edges_full.sum(axis=0)
    profile_v_smooth = gaussian_filter1d(vertical_edge_profile, sigma=20)
    
    peaks_v, _ = find_peaks(profile_v_smooth,
                           height=np.percentile(profile_v_smooth, 75),
                           distance=300)
    
    if len(peaks_v) < 2:
        # Fallback: use center region
        if debug:
            print(f"  ⚠ Using default horizontal range (couldn't find ruler edges)")
        x_start = x_min
        x_end = x_max
    else:
        # Take outermost peaks
        x_start = peaks_v[0]
        x_end = peaks_v[-1]
    
    if debug:
        print(f"  ✓ Ruler region: x={x_start} to x={x_end}, y={y_start} to y={y_end}")
        print(f"    Width: {x_end-x_start}px, Height: {y_end-y_start}px")
    
    return (x_start, x_end, y_start, y_end)


def measure_ruler_ticks_top(strip: np.ndarray,
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
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    
    # Extract ruler region
    ruler_region = gray[y_start:y_end, x_start:x_end]
    
    # Average vertically to get horizontal intensity profile
    profile = ruler_region.mean(axis=0)
    
    # Smooth profile
    profile_smooth = gaussian_filter1d(profile, sigma=3)
    
    # Ruler tick marks are typically darker than background
    # Invert profile to find dark marks as peaks
    profile_inv = 255 - profile_smooth
    
    # Find peaks (tick marks)
    threshold = np.percentile(profile_inv, 65)
    
    peaks, properties = find_peaks(profile_inv,
                                   height=threshold,
                                   distance=50)  # Minimum distance between major ticks
    
    if len(peaks) < 5:
        # Try with lower threshold
        threshold = np.percentile(profile_inv, 55)
        peaks, properties = find_peaks(profile_inv,
                                       height=threshold,
                                       distance=50)
    
    if len(peaks) < 5:
        if debug:
            print(f"  ⚠ Only found {len(peaks)} tick marks (need at least 5)")
        return None
    
    if debug:
        print(f"  ✓ Found {len(peaks)} tick marks")
    
    # Calculate spacing between consecutive ticks
    spacings = np.diff(peaks)
    
    # Remove outliers (filter for major ticks only)
    median_spacing = np.median(spacings)
    # Keep spacings within 40% of median (to filter out minor ticks)
    valid_spacings = spacings[np.abs(spacings - median_spacing) < 0.4 * median_spacing]
    
    if len(valid_spacings) < 3:
        if debug:
            print(f"  ⚠ Not enough valid spacings after outlier removal")
        # Try being more lenient
        valid_spacings = spacings[np.abs(spacings - median_spacing) < 0.6 * median_spacing]
        if len(valid_spacings) < 3:
            return None
    
    avg_spacing = np.mean(valid_spacings)
    std_spacing = np.std(valid_spacings)
    
    if debug:
        print(f"  Tick spacing: {avg_spacing:.1f} ± {std_spacing:.1f} px")
        print(f"  Valid spacings: {len(valid_spacings)}/{len(spacings)}")
        print(f"  All spacings: {spacings}")
    
    # Calculate calibration
    # For a 10mm ruler with N peaks, assume (N-1) intervals
    # If we have ~11 peaks for 10mm ruler = 1mm per interval
    mm_per_tick = ruler_mm / (len(peaks) - 1)
    um_per_tick = mm_per_tick * 1000.0
    um_per_pixel = um_per_tick / avg_spacing
    
    if debug:
        print(f"  Calculated: {mm_per_tick:.2f} mm/tick = {um_per_tick:.1f} µm/tick")
        print(f"  Calibration: {um_per_pixel:.3f} µm/pixel")
    
    return {
        'num_ticks': len(peaks),
        'spacing_px': float(avg_spacing),
        'spacing_std_px': float(std_spacing),
        'mm_per_tick': float(mm_per_tick),
        'um_per_tick': float(um_per_tick),
        'um_per_pixel': float(um_per_pixel),
        'ruler_x_start': int(x_start),
        'ruler_x_end': int(x_end),
        'ruler_y_start': int(y_start),
        'ruler_y_end': int(y_end),
    }


def auto_calibrate_ruler_top(image_path: Path,
                             ruler_mm: float = 10.0,
                             strip_height: int = 500,
                             debug: bool = False) -> Optional[Dict]:
    """
    Automatically detect and calibrate ruler in TOP of image.
    
    Returns:
        Calibration dict or None if ruler not found
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    print(f"\nExtracting top strip (height={strip_height}px)...")
    strip = extract_top_strip(img, strip_height)
    
    print(f"Searching for ruler in top center region...")
    ruler_region = find_ruler_region_top(strip, debug=debug)
    
    if ruler_region is None:
        print("❌ Could not find ruler region")
        return None
    
    x_start, x_end, y_start, y_end = ruler_region
    print(f"✓ Ruler found at x={x_start}-{x_end}, y={y_start}-{y_end}")
    
    print(f"\nMeasuring tick spacing...")
    result = measure_ruler_ticks_top(strip, x_start, x_end, y_start, y_end, ruler_mm, debug=debug)
    
    if result is None:
        print("❌ Could not measure ruler tick spacing")
        return None
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC CALIBRATION RESULT (TOP)")
    print(f"{'='*70}")
    print(f"Ruler location: Top center")
    print(f"Ruler position: x={result['ruler_x_start']}-{result['ruler_x_end']}, y={result['ruler_y_start']}-{result['ruler_y_end']}")
    print(f"Tick marks detected: {result['num_ticks']}")
    print(f"Average spacing: {result['spacing_px']:.1f} ± {result['spacing_std_px']:.1f} px")
    print(f"Assumed: {result['mm_per_tick']:.2f} mm per tick = {result['um_per_tick']:.1f} µm")
    print(f"→ Calibration: {result['um_per_pixel']:.3f} µm/pixel")
    print(f"{'='*70}")
    
    return {
        'image': str(image_path),
        'ruler_mm': float(ruler_mm),
        'ruler_location': 'top_center',
        'strip_height': int(strip_height),
        'num_ticks': int(result['num_ticks']),
        'spacing_px': float(result['spacing_px']),
        'spacing_std_px': float(result['spacing_std_px']),
        'mm_per_tick': float(result['mm_per_tick']),
        'um_per_tick': float(result['um_per_tick']),
        'um_per_pixel': float(result['um_per_pixel']),
        'ruler_x_start': int(result['ruler_x_start']),
        'ruler_x_end': int(result['ruler_x_end']),
        'ruler_y_start': int(result['ruler_y_start']),
        'ruler_y_end': int(result['ruler_y_end']),
        'method': 'automatic_top_center'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Automatic ruler detection (TOP of image)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image with ruler')
    parser.add_argument('--ruler-mm', type=float, default=10.0,
                        help='Ruler length in millimeters (default: 10.0)')
    parser.add_argument('--strip-height', type=int, default=500,
                        help='Height of top strip to analyze (default: 500)')
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
    result = auto_calibrate_ruler_top(
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
        output_path = output_dir / f"{image_path.stem}_top_calibration.json"
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
