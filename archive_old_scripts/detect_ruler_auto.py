#!/usr/bin/env python3
"""
Automatic ruler detection and calibration for microscope images.

This script automatically:
1. Searches all 4 corners for ruler presence
2. Detects ruler edge/markings using edge detection
3. Measures ruler spacing between major tick marks
4. Calculates µm/pixel calibration

Usage:
    python scripts/detect_ruler_auto.py \
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
from PIL import Image, ImageDraw
import cv2


def detect_ruler_in_corner(crop: np.ndarray, 
                           corner_name: str,
                           debug: bool = False) -> Optional[Dict]:
    """
    Detect ruler in a corner crop using edge detection.
    
    Returns:
        Dict with ruler info or None if no ruler found
    """
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=100, 
                            minLineLength=200, 
                            maxLineGap=20)
    
    if lines is None or len(lines) == 0:
        return None
    
    # Find dominant horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        
        # Horizontal line (angle close to 0 or π)
        if angle < np.pi/6 or angle > 5*np.pi/6:
            h_lines.append(line[0])
        # Vertical line (angle close to π/2)
        elif np.pi/3 < angle < 2*np.pi/3:
            v_lines.append(line[0])
    
    # Rulers typically have strong horizontal/vertical lines
    has_ruler = len(h_lines) > 2 or len(v_lines) > 2
    
    if not has_ruler:
        return None
    
    # Estimate ruler orientation
    if len(h_lines) > len(v_lines):
        orientation = 'horizontal'
        main_lines = h_lines
    else:
        orientation = 'vertical'
        main_lines = v_lines
    
    if debug:
        print(f"{corner_name}: Found {len(h_lines)} H-lines, {len(v_lines)} V-lines")
        print(f"  Orientation: {orientation}")
    
    return {
        'corner': corner_name,
        'orientation': orientation,
        'num_lines': len(main_lines),
        'edges_density': edges.sum() / edges.size,
    }


def find_ruler_spacing(crop: np.ndarray, 
                       orientation: str,
                       expected_ticks: int = 10) -> Optional[float]:
    """
    Find spacing between ruler tick marks.
    
    Args:
        crop: Grayscale image of ruler region
        orientation: 'horizontal' or 'vertical'
        expected_ticks: Expected number of major tick marks
    
    Returns:
        Average spacing in pixels or None
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    
    # Get intensity profile along ruler direction
    if orientation == 'horizontal':
        # Average over vertical direction to reduce noise
        profile = gray.mean(axis=0)
    else:
        # Average over horizontal direction
        profile = gray.mean(axis=1)
    
    # Find peaks (tick marks are darker or brighter than background)
    # Try both dark and bright marks
    profile_inv = 255 - profile
    
    # Smooth profile
    from scipy.ndimage import gaussian_filter1d
    profile_smooth = gaussian_filter1d(profile, sigma=3)
    profile_inv_smooth = gaussian_filter1d(profile_inv, sigma=3)
    
    # Find peaks
    from scipy.signal import find_peaks
    
    # Try dark marks
    peaks_dark, _ = find_peaks(profile_inv_smooth, 
                               height=np.percentile(profile_inv_smooth, 70),
                               distance=30)
    
    # Try bright marks
    peaks_bright, _ = find_peaks(profile_smooth,
                                 height=np.percentile(profile_smooth, 70),
                                 distance=30)
    
    # Use whichever gives more reasonable number of peaks
    peaks = peaks_dark if len(peaks_dark) >= len(peaks_bright) else peaks_bright
    
    if len(peaks) < 2:
        return None
    
    # Calculate average spacing between consecutive peaks
    spacings = np.diff(peaks)
    avg_spacing = np.median(spacings)  # Use median to ignore outliers
    
    return float(avg_spacing)


def auto_calibrate_ruler(image_path: Path,
                          ruler_mm: float = 10.0,
                          corner_size: int = 1500,
                          debug: bool = False) -> Optional[Dict]:
    """
    Automatically detect and calibrate ruler.
    
    Returns:
        Calibration dict or None if ruler not found
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Define corners to check
    corners = {
        'top_left': (0, 0, corner_size, corner_size),
        'top_right': (img.width - corner_size, 0, img.width, corner_size),
        'bottom_left': (0, img.height - corner_size, corner_size, img.height),
        'bottom_right': (img.width - corner_size, img.height - corner_size, 
                        img.width, img.height),
    }
    
    print(f"\nSearching for ruler in corners...")
    
    # Check each corner
    ruler_found = None
    best_score = 0
    
    for name, bbox in corners.items():
        crop_pil = img.crop(bbox)
        crop = np.array(crop_pil.convert('RGB'))
        
        result = detect_ruler_in_corner(crop, name, debug=debug)
        
        if result is not None:
            # Score based on number of lines and edge density
            score = result['num_lines'] * result['edges_density']
            
            if debug:
                print(f"  {name}: score={score:.2f}")
            
            if score > best_score:
                best_score = score
                ruler_found = {
                    'name': name,
                    'bbox': bbox,
                    'crop': crop,
                    'info': result
                }
    
    if ruler_found is None:
        print("❌ No ruler detected in any corner")
        return None
    
    print(f"✓ Ruler detected in: {ruler_found['name']}")
    print(f"  Orientation: {ruler_found['info']['orientation']}")
    
    # Measure ruler spacing
    spacing = find_ruler_spacing(
        ruler_found['crop'],
        ruler_found['info']['orientation'],
        expected_ticks=10
    )
    
    if spacing is None:
        print("❌ Could not measure ruler spacing")
        return None
    
    print(f"✓ Detected spacing: {spacing:.1f} pixels")
    
    # Calculate calibration
    # Assume ruler has major ticks every 1mm (10 ticks for 10mm ruler)
    mm_per_tick = ruler_mm / 10.0  # 1mm per major tick
    um_per_tick = mm_per_tick * 1000.0  # Convert to µm
    
    um_per_pixel = um_per_tick / spacing
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC CALIBRATION RESULT")
    print(f"{'='*70}")
    print(f"Ruler location: {ruler_found['name']}")
    print(f"Ruler orientation: {ruler_found['info']['orientation']}")
    print(f"Detected spacing: {spacing:.1f} px")
    print(f"Assumed tick spacing: {mm_per_tick} mm = {um_per_tick} µm")
    print(f"Calibration: {um_per_pixel:.3f} µm/pixel")
    print(f"{'='*70}")
    
    return {
        'image': str(image_path),
        'ruler_mm': ruler_mm,
        'ruler_location': ruler_found['name'],
        'ruler_orientation': ruler_found['info']['orientation'],
        'spacing_px': spacing,
        'mm_per_tick': mm_per_tick,
        'um_per_tick': um_per_tick,
        'um_per_pixel': um_per_pixel,
        'method': 'automatic_edge_detection'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Automatic ruler detection and calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic detection with 10mm ruler
  python scripts/detect_ruler_auto.py \
      --image data/slike/K1_Fe2O3001_(1).jpg \
      --ruler-mm 10

  # Save calibration to specific location
  python scripts/detect_ruler_auto.py \
      --image plate.jpg \
      --ruler-mm 10 \
      --output calibration/plate_cal.json

  # Debug mode (verbose output)
  python scripts/detect_ruler_auto.py \
      --image plate.jpg \
      --ruler-mm 10 \
      --debug
        """
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
    result = auto_calibrate_ruler(
        image_path=image_path,
        ruler_mm=args.ruler_mm,
        debug=args.debug
    )
    
    if result is None:
        print("\n❌ Automatic calibration failed")
        print("Fallback: Use manual calibration (calibrate_ruler.py)")
        sys.exit(1)
    
    # Save calibration
    if args.output is None:
        output_path = Path('data/calibration') / f"{image_path.stem}_auto_calibration.json"
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
