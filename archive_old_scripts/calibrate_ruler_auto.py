#!/usr/bin/env python3
"""
Automatic ruler calibration with WHITE tick detection.

Extracts ruler region (x=4000-9000, y=1500-3000), detects white ticks,
and calculates µm/pixel calibration based on 5mm major tick intervals.

Usage:
    python scripts/calibrate_ruler_auto.py --image "data/slike/K1_Fe2O3001 (1).jpg"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Fixed ruler region coordinates (same across all images)
RULER_X_START = 4000
RULER_X_END = 9000
RULER_Y_START = 1500
RULER_Y_END = 3000
MAJOR_TICK_INTERVAL_MM = 5.0  # 0.5 cm


def calibrate_ruler_auto(image_path: Path, output_dir: Path = Path('output')) -> dict:
    """
    Automatically calibrate ruler from image.
    
    Returns:
        Calibration dict with um_per_pixel and metadata
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Extract ruler region
    print(f"\nExtracting ruler region...")
    print(f"  x={RULER_X_START}-{RULER_X_END}, y={RULER_Y_START}-{RULER_Y_END}")
    
    ruler_img = img.crop((RULER_X_START, RULER_Y_START, RULER_X_END, RULER_Y_END))
    ruler_np = np.array(ruler_img.convert('RGB'))
    gray = cv2.cvtColor(ruler_np, cv2.COLOR_RGB2GRAY)
    
    print(f"  Ruler region: {ruler_np.shape[1]} × {ruler_np.shape[0]} pixels")
    
    # Get intensity profile
    profile = gray.mean(axis=0)
    profile_smooth = gaussian_filter1d(profile, sigma=5)
    
    # Detect MAJOR ticks (WHITE/bright, 5mm intervals)
    threshold_major = np.percentile(profile_smooth, 80)
    peaks_major, _ = find_peaks(profile_smooth,
                                height=threshold_major,
                                distance=300,
                                prominence=20)
    
    # Detect ALL ticks (including minor)
    threshold_all = np.percentile(profile_smooth, 65)
    peaks_all, _ = find_peaks(profile_smooth,
                              height=threshold_all,
                              distance=50,
                              prominence=5)
    
    # Classify as major or minor
    major_set = set()
    minor_set = set()
    
    for peak in peaks_all:
        is_major = False
        for major_peak in peaks_major:
            if abs(peak - major_peak) < 30:
                major_set.add(major_peak)
                is_major = True
                break
        if not is_major:
            minor_set.add(peak)
    
    peaks_major = sorted(list(major_set))
    peaks_minor = sorted(list(minor_set))
    
    print(f"\nTick detection (WHITE ticks):")
    print(f"  Major ticks: {len(peaks_major)}")
    print(f"  Minor ticks: {len(peaks_minor)}")
    print(f"  Total ticks: {len(peaks_major) + len(peaks_minor)}")
    
    # Validate detection
    if len(peaks_major) < 5:
        print(f"\n⚠ WARNING: Only {len(peaks_major)} major ticks detected (expected ≥5)")
        print(f"Using fallback calibration: 8.666 µm/pixel")
        return create_fallback_calibration(image_path, output_dir)
    
    # Calculate spacing
    spacings_major = np.diff(peaks_major)
    avg_spacing = np.mean(spacings_major)
    std_spacing = np.std(spacings_major)
    variation_pct = (std_spacing / avg_spacing) * 100
    
    print(f"\nMajor tick spacing:")
    print(f"  Mean: {avg_spacing:.1f} ± {std_spacing:.1f} px")
    print(f"  Variation: {variation_pct:.1f}%")
    
    # Check spacing consistency
    if variation_pct > 15:
        print(f"\n⚠ WARNING: High spacing variation ({variation_pct:.1f}%)")
        print(f"Calibration may be inaccurate. Using fallback: 8.666 µm/pixel")
        return create_fallback_calibration(image_path, output_dir)
    
    # Calculate calibration
    um_per_interval = MAJOR_TICK_INTERVAL_MM * 1000.0
    um_per_pixel = um_per_interval / avg_spacing
    total_ruler_mm = (len(peaks_major) - 1) * MAJOR_TICK_INTERVAL_MM
    
    print(f"\n{'='*70}")
    print(f"CALIBRATION RESULT")
    print(f"{'='*70}")
    print(f"Major ticks detected: {len(peaks_major)} (5mm intervals)")
    print(f"Average spacing: {avg_spacing:.1f} px")
    print(f"→ Calibration: {um_per_pixel:.3f} µm/pixel")
    print(f"Ruler coverage: {total_ruler_mm} mm ({total_ruler_mm/10:.1f} cm)")
    print(f"{'='*70}")
    
    # Validate minor tick spacing
    minor_validation = ""
    if len(peaks_minor) >= 2:
        spacings_minor = np.diff(peaks_minor)
        avg_minor_spacing = np.mean(spacings_minor)
        minor_interval_mm = (MAJOR_TICK_INTERVAL_MM * avg_minor_spacing) / avg_spacing
        minor_validation = f"Minor tick interval: ~{minor_interval_mm:.2f} mm"
        print(f"\nValidation: {minor_validation}")
    
    # Create calibration dict
    calibration = {
        'image': str(image_path),
        'ruler_type': '10cm ruler with 5mm (0.5cm) WHITE major ticks',
        'ruler_x_start': RULER_X_START,
        'ruler_x_end': RULER_X_END,
        'ruler_y_start': RULER_Y_START,
        'ruler_y_end': RULER_Y_END,
        'num_major_ticks': int(len(peaks_major)),
        'num_minor_ticks': int(len(peaks_minor)),
        'major_tick_positions': [int(p) for p in peaks_major],
        'minor_tick_positions': [int(p) for p in peaks_minor],
        'major_spacings': [int(s) for s in spacings_major],
        'mean_spacing_px': float(avg_spacing),
        'std_spacing_px': float(std_spacing),
        'variation_percent': float(variation_pct),
        'mm_per_interval': float(MAJOR_TICK_INTERVAL_MM),
        'um_per_interval': float(um_per_interval),
        'um_per_pixel': float(um_per_pixel),
        'ruler_length_mm': float(total_ruler_mm),
        'minor_validation': minor_validation,
        'method': 'automatic_white_ticks',
        'status': 'success'
    }
    
    # Generate visualization
    generate_calibration_visualization(
        ruler_np, profile_smooth, peaks_major, peaks_minor,
        spacings_major, calibration, image_path, output_dir
    )
    
    # Save calibration JSON
    image_name = image_path.stem
    calib_path = output_dir / f"{image_name}_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✓ Saved: {calib_path}")
    
    return calibration


def create_fallback_calibration(image_path: Path, output_dir: Path) -> dict:
    """Create fallback calibration using default value."""
    calibration = {
        'image': str(image_path),
        'um_per_pixel': 8.666,
        'method': 'fallback_default',
        'status': 'fallback',
        'note': 'Automatic detection failed, using default calibration'
    }
    
    image_name = image_path.stem
    calib_path = output_dir / f"{image_name}_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✓ Saved fallback calibration: {calib_path}")
    
    return calibration


def generate_calibration_visualization(ruler_np, profile_smooth, peaks_major, peaks_minor,
                                      spacings_major, calibration, image_path, output_dir):
    """Generate ruler calibration visualization."""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)
    
    # Panel 1: Ruler with ticks
    ax1 = fig.add_subplot(gs[0, 0])
    ruler_display = ruler_np.copy()
    
    # Draw minor ticks (yellow)
    for peak in peaks_minor:
        cv2.line(ruler_display, (peak, 0), (peak, ruler_np.shape[0]), (255, 255, 0), 2)
    
    # Draw major ticks (green)
    for i, peak in enumerate(peaks_major):
        cv2.line(ruler_display, (peak, 0), (peak, ruler_np.shape[0]), (0, 255, 0), 5)
        cm_value = i * 0.5
        cv2.putText(ruler_display, f'{cm_value:.1f}', (peak-35, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    
    ax1.imshow(ruler_display)
    ax1.set_title(f'Ruler: {len(peaks_major)} Major (GREEN) + {len(peaks_minor)} Minor (YELLOW) Ticks',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position (pixels)')
    
    # Panel 2: Intensity profile
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(profile_smooth, linewidth=2, color='blue', label='Intensity (bright = ticks)')
    for peak in peaks_minor:
        ax2.plot(peak, profile_smooth[peak], 'yo', markersize=6)
    for peak in peaks_major:
        ax2.plot(peak, profile_smooth[peak], 'go', markersize=12)
    ax2.set_xlabel('Position (pixels)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Intensity Profile - WHITE Ticks')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Spacing
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(range(len(spacings_major)), spacings_major, color='green', alpha=0.7)
    ax3.axhline(y=calibration['mean_spacing_px'], color='red', linestyle='--',
               label=f"Mean: {calibration['mean_spacing_px']:.1f} px")
    ax3.set_xlabel('Interval index')
    ax3.set_ylabel('Spacing (pixels)')
    ax3.set_title(f"Major Tick Spacing (Variation: {calibration['variation_percent']:.1f}%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Ruler Calibration: {calibration['um_per_pixel']:.3f} µm/pixel",
                fontsize=16, fontweight='bold')
    
    image_name = image_path.stem
    viz_path = output_dir / f"{image_name}_ruler_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='Automatic ruler calibration')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calibration = calibrate_ruler_auto(image_path, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Calibration: {calibration['um_per_pixel']:.3f} µm/pixel")
    print(f"Status: {calibration['status']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
