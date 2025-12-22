#!/usr/bin/env python3
"""
Visual debug script to visualize ruler detection process.

Saves annotated images showing:
- Bottom strip extraction
- Detected ruler boundaries
- Detected tick marks
- Intensity profile

Usage:
    python scripts/debug_ruler_detection.py --image "data/slike/K1_Fe2O3001 (1).jpg"
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def debug_ruler_detection(image_path: Path, 
                          strip_height: int = 300,
                          output_dir: Path = Path('ruler_debug')):
    """
    Run ruler detection with full visualization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Extract bottom strip
    print(f"\n1. Extracting bottom strip (height={strip_height}px)...")
    bottom_strip_pil = img.crop((0, img.height - strip_height, img.width, img.height))
    strip = np.array(bottom_strip_pil.convert('RGB'))
    
    # Save strip
    strip_path = output_dir / "1_bottom_strip.jpg"
    bottom_strip_pil.save(strip_path, quality=95)
    print(f"   Saved: {strip_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Edge detection
    print(f"\n2. Edge detection...")
    edges = cv2.Canny(gray, 50, 150)
    edges_path = output_dir / "2_edges.jpg"
    cv2.imwrite(str(edges_path), edges)
    print(f"   Saved: {edges_path}")
    
    # Vertical edge profile
    print(f"\n3. Finding ruler boundaries...")
    vertical_edge_profile = edges.sum(axis=0)
    profile_smooth = gaussian_filter1d(vertical_edge_profile, sigma=20)
    
    # Plot profile
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(vertical_edge_profile, alpha=0.3, label='Raw')
    ax.plot(profile_smooth, label='Smoothed (σ=20)', linewidth=2)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Edge Density')
    ax.set_title('Vertical Edge Profile (for finding ruler boundaries)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    profile_path = output_dir / "3_vertical_profile.png"
    plt.savefig(profile_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {profile_path}")
    
    # Find peaks
    peaks, properties = find_peaks(profile_smooth, 
                                   height=np.percentile(profile_smooth, 80),
                                   distance=500)
    
    print(f"   Found {len(peaks)} ruler boundary candidates")
    if len(peaks) >= 2:
        x_start = peaks[0]
        x_end = peaks[1]
        print(f"   Using: x={x_start} to x={x_end} (width={x_end-x_start}px)")
        
        # Annotate strip with boundaries
        strip_annotated = strip.copy()
        cv2.line(strip_annotated, (x_start, 0), (x_start, h), (0, 255, 0), 3)
        cv2.line(strip_annotated, (x_end, 0), (x_end, h), (0, 255, 0), 3)
        cv2.putText(strip_annotated, f"Ruler Region: {x_end-x_start}px", 
                   (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        boundaries_path = output_dir / "4_ruler_boundaries.jpg"
        cv2.imwrite(str(boundaries_path), cv2.cvtColor(strip_annotated, cv2.COLOR_RGB2BGR))
        print(f"   Saved: {boundaries_path}")
        
        # Extract ruler region
        print(f"\n4. Analyzing ruler ticks...")
        ruler_region = gray[:, x_start:x_end]
        
        # Horizontal intensity profile
        profile = ruler_region.mean(axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=3)
        profile_inv = 255 - profile_smooth
        
        # Plot horizontal profile
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
        
        # Top: ruler region
        ax1.imshow(ruler_region, cmap='gray', aspect='auto')
        ax1.set_title('Ruler Region (grayscale)')
        ax1.set_ylabel('Y (pixels)')
        
        # Bottom: intensity profile
        ax2.plot(profile, alpha=0.3, label='Raw intensity')
        ax2.plot(profile_smooth, label='Smoothed', linewidth=2)
        ax2.plot(profile_inv, label='Inverted (for dark ticks)', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Position along ruler (pixels)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Horizontal Intensity Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        profile_h_path = output_dir / "5_ruler_profile.png"
        plt.savefig(profile_h_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {profile_h_path}")
        
        # Find tick marks
        threshold = np.percentile(profile_inv, 60)
        peaks_ticks, _ = find_peaks(profile_inv, height=threshold, distance=30)
        
        print(f"   Found {len(peaks_ticks)} tick marks")
        print(f"   Tick positions: {peaks_ticks}")
        
        if len(peaks_ticks) >= 2:
            spacings = np.diff(peaks_ticks)
            median_spacing = np.median(spacings)
            print(f"   Spacings: {spacings}")
            print(f"   Median spacing: {median_spacing:.1f} px")
            
            # Annotate ticks on ruler
            ruler_rgb = cv2.cvtColor(ruler_region, cv2.COLOR_GRAY2RGB)
            for i, peak in enumerate(peaks_ticks):
                cv2.line(ruler_rgb, (peak, 0), (peak, h), (255, 0, 0), 2)
                cv2.putText(ruler_rgb, str(i), (peak-10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            ticks_path = output_dir / "6_detected_ticks.jpg"
            cv2.imwrite(str(ticks_path), cv2.cvtColor(ruler_rgb, cv2.COLOR_RGB2BGR))
            print(f"   Saved: {ticks_path}")
    
    print(f"\n{'='*70}")
    print(f"DEBUG OUTPUT COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  1_bottom_strip.jpg      - Extracted bottom strip")
    print(f"  2_edges.jpg             - Canny edge detection")
    print(f"  3_vertical_profile.png  - Profile for finding ruler boundaries")
    print(f"  4_ruler_boundaries.jpg  - Detected ruler region")
    print(f"  5_ruler_profile.png     - Intensity profile along ruler")
    print(f"  6_detected_ticks.jpg    - Tick mark detection")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Debug ruler detection with visualizations',
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image')
    parser.add_argument('--strip-height', type=int, default=300,
                        help='Height of bottom strip (default: 300)')
    parser.add_argument('--output', type=str, default='ruler_debug',
                        help='Output directory (default: ruler_debug)')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    debug_ruler_detection(
        image_path=image_path,
        strip_height=args.strip_height,
        output_dir=Path(args.output)
    )


if __name__ == '__main__':
    main()
