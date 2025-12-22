#!/usr/bin/env python3
"""
Create a comprehensive visual report of ruler detection.

Generates a multi-panel figure showing:
- Original bottom strip
- Detected ruler region with boundaries
- Detected tick marks
- Intensity profile with peaks marked

Usage:
    python scripts/visualize_ruler_detection.py --image "data/slike/K1_Fe2O3001 (1).jpg"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_ruler_visualization(image_path: Path, 
                               strip_height: int = 500,
                               location: str = 'top',
                               output_path: Path = Path('ruler_detection_visualization.png')):
    """
    Create comprehensive visualization of ruler detection.
    
    Args:
        location: 'top' or 'bottom' - where to look for ruler
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Extract strip based on location
    if location == 'top':
        print(f"Extracting TOP strip (height={strip_height}px)...")
        strip_pil = img.crop((0, 0, img.width, strip_height))
    else:
        print(f"Extracting BOTTOM strip (height={strip_height}px)...")
        strip_pil = img.crop((0, img.height - strip_height, img.width, img.height))
    
    strip = np.array(strip_pil.convert('RGB'))
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Edge detection for finding ruler boundaries
    print(f"Finding ruler boundaries...")
    edges = cv2.Canny(gray, 50, 150)
    vertical_edge_profile = edges.sum(axis=0)
    profile_smooth = gaussian_filter1d(vertical_edge_profile, sigma=20)
    
    # Find ruler boundaries
    peaks_boundaries, _ = find_peaks(profile_smooth, 
                                     height=np.percentile(profile_smooth, 80),
                                     distance=500)
    
    if len(peaks_boundaries) < 2:
        print("ERROR: Could not find ruler boundaries")
        return
    
    x_start = peaks_boundaries[0]
    x_end = peaks_boundaries[1]
    print(f"Ruler region: x={x_start} to x={x_end} (width={x_end-x_start}px)")
    
    # Extract ruler region
    print(f"Detecting tick marks...")
    ruler_region = gray[:, x_start:x_end]
    
    # Horizontal intensity profile
    profile = ruler_region.mean(axis=0)
    profile_smooth_h = gaussian_filter1d(profile, sigma=3)
    profile_inv = 255 - profile_smooth_h
    
    # Find tick marks
    threshold = np.percentile(profile_inv, 60)
    peaks_ticks, properties = find_peaks(profile_inv, height=threshold, distance=30)
    
    print(f"Found {len(peaks_ticks)} tick marks")
    print(f"Tick positions (relative to ruler start): {peaks_ticks}")
    
    spacings = np.array([])
    if len(peaks_ticks) >= 2:
        spacings = np.diff(peaks_ticks)
        print(f"Spacings between consecutive ticks: {spacings}")
        print(f"Mean spacing: {np.mean(spacings):.1f} px")
        print(f"Median spacing: {np.median(spacings):.1f} px")
    
    # Create figure
    print(f"\nCreating visualization...")
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Full strip
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(strip)
    ax1.axvline(x=x_start, color='lime', linewidth=3, linestyle='--', label='Ruler boundaries')
    ax1.axvline(x=x_end, color='lime', linewidth=3, linestyle='--')
    ax1.set_title(f'{location.upper()} Strip of Image with Detected Ruler Boundaries', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.legend(fontsize=12)
    
    # Panel 2: Vertical edge profile (for boundary detection)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(vertical_edge_profile, alpha=0.3, label='Raw edge profile', color='gray')
    ax2.plot(profile_smooth, label='Smoothed profile', linewidth=2, color='blue')
    ax2.axvline(x=x_start, color='lime', linewidth=2, linestyle='--', alpha=0.7)
    ax2.axvline(x=x_end, color='lime', linewidth=2, linestyle='--', alpha=0.7)
    for peak in peaks_boundaries[:10]:  # Show first 10 peaks
        ax2.plot(peak, profile_smooth[peak], 'ro', markersize=8)
    ax2.set_title('Vertical Edge Profile (used to find ruler boundaries)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Edge Density')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Ruler region with tick marks
    ax3 = fig.add_subplot(gs[2, 0])
    ruler_rgb = cv2.cvtColor(ruler_region, cv2.COLOR_GRAY2RGB)
    for i, peak in enumerate(peaks_ticks):
        cv2.line(ruler_rgb, (peak, 0), (peak, h), (255, 0, 0), 2)
        cv2.putText(ruler_rgb, str(i), (peak-8, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    ax3.imshow(ruler_rgb)
    ax3.set_title(f'Detected Ruler Region with {len(peaks_ticks)} Tick Marks', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Position along ruler (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # Panel 4: Horizontal intensity profile with peaks
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(profile, alpha=0.4, label='Raw intensity', color='gray')
    ax4.plot(profile_smooth_h, label='Smoothed', linewidth=2, color='blue')
    ax4.plot(profile_inv, label='Inverted (for dark ticks)', linewidth=2, color='orange', alpha=0.7)
    ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=1, 
               label=f'Threshold ({threshold:.1f})')
    for peak in peaks_ticks:
        ax4.plot(peak, profile_inv[peak], 'ro', markersize=8)
    ax4.set_title('Horizontal Intensity Profile (tick detection)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Position along ruler (pixels)')
    ax4.set_ylabel('Intensity')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Spacing analysis
    ax5 = fig.add_subplot(gs[3, 0])
    if len(peaks_ticks) >= 2:
        ax5.bar(range(len(spacings)), spacings, color='steelblue', edgecolor='black')
        ax5.axhline(y=np.mean(spacings), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(spacings):.1f} px')
        ax5.axhline(y=np.median(spacings), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(spacings):.1f} px')
        ax5.set_title('Spacing Between Consecutive Ticks', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Tick pair index')
        ax5.set_ylabel('Spacing (pixels)')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, spacing in enumerate(spacings):
            ax5.text(i, spacing + 2, f'{spacing:.0f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 6: Summary statistics
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    # Calculate calibration (assuming 10mm ruler)
    ruler_mm = 10.0
    mm_per_tick = ruler_mm / (len(peaks_ticks) - 1) if len(peaks_ticks) > 1 else 0
    um_per_tick = mm_per_tick * 1000.0
    avg_spacing = np.mean(spacings) if len(spacings) > 0 else 0
    um_per_pixel = um_per_tick / avg_spacing if avg_spacing > 0 else 0
    
    summary_text = f"""
RULER DETECTION SUMMARY
{'='*50}

Image: {image_path.name}
Image size: {img.width} × {img.height} pixels

RULER LOCATION:
  Position: Bottom edge
  X range: {x_start} to {x_end} pixels
  Width: {x_end - x_start} pixels

TICK DETECTION:
  Number of ticks detected: {len(peaks_ticks)}
  Tick positions: {list(peaks_ticks)}
  
SPACING ANALYSIS:
  Spacings: {list(spacings) if len(spacings) > 0 else 'N/A'}
  Mean spacing: {avg_spacing:.1f} ± {np.std(spacings):.1f} px
  Median spacing: {np.median(spacings):.1f} px
  
CALIBRATION (assuming {ruler_mm}mm ruler):
  mm per tick: {mm_per_tick:.3f} mm
  µm per tick: {um_per_tick:.1f} µm
  → µm per pixel: {um_per_pixel:.3f} µm/px

NOTE: Irregular spacings suggest both major
and minor ticks may be detected. Manual
verification recommended.
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Main title
    fig.suptitle('AUTOMATIC RULER DETECTION VISUALIZATION', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    plt.close()
    
    return {
        'num_ticks': len(peaks_ticks),
        'spacings': spacings if len(spacings) > 0 else [],
        'um_per_pixel': um_per_pixel,
        'ruler_x_start': x_start,
        'ruler_x_end': x_end,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive ruler detection visualization',
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image')
    parser.add_argument('--location', type=str, default='top', choices=['top', 'bottom'],
                        help='Where to look for ruler: top or bottom (default: top)')
    parser.add_argument('--strip-height', type=int, default=500,
                        help='Height of strip (default: 500)')
    parser.add_argument('--output', type=str, default='ruler_detection_visualization.png',
                        help='Output image path (default: ruler_detection_visualization.png)')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    result = create_ruler_visualization(
        image_path=image_path,
        strip_height=args.strip_height,
        location=args.location,
        output_path=Path(args.output)
    )
    
    if result:
        print(f"\n{'='*70}")
        print(f"VISUALIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nKey findings:")
        print(f"  Ticks detected: {result['num_ticks']}")
        print(f"  Ruler position: x={result['ruler_x_start']} to x={result['ruler_x_end']}")
        print(f"  Calibration: {result['um_per_pixel']:.3f} µm/pixel")
        print(f"\nNext step: Open {args.output} to visually confirm the detection")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
