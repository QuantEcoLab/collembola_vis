#!/usr/bin/env python3
"""
Visualize organism measurements with overlay annotations.

Creates overview grid showing all detected organisms with measurements.

Usage:
    python scripts/visualize_measurements.py \
      --image "data/slike/K1.jpg" \
      --detections detections.csv \
      --measurements measurements.csv \
      --output visualizations/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_measurement_visualization(image_path: Path,
                                     detections_csv: Path,
                                     measurements_csv: Path,
                                     output_dir: Path,
                                     num_samples: int = 50):
    """
    Create visualization of measurements overlaid on organisms.
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    
    print(f"Loading detections: {detections_csv}")
    detections = pd.read_csv(detections_csv)
    detections['detection_id'] = range(len(detections))  # Add sequential ID
    
    print(f"Loading measurements: {measurements_csv}")
    measurements = pd.read_csv(measurements_csv)
    
    print(f"\nDataset: {len(measurements)} organisms")
    
    # Merge detections and measurements
    data = pd.merge(detections, measurements, on='detection_id', suffixes=('_det', '_meas'))
    
    # Create overview visualization (all organisms)
    create_overview(img, data, image_path, output_dir)
    
    # Create samples visualization (random subset with details)
    create_samples(img, data, image_path, output_dir, num_samples)
    
    print(f"\n✓ Visualization complete")


def create_overview(img, data, image_path, output_dir):
    """Create overview image with all bounding boxes and labels."""
    print(f"\nCreating overview visualization...")
    
    # Create copy for drawing
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Draw each organism
    for idx, row in data.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        length = row['length_mm']
        width = row['width_mm']
        conf = row['confidence_det']  # Use detection confidence (from detections CSV)
        
        # Color code by size (green=small, yellow=medium, red=large)
        # Using mm: small <0.5mm, medium 0.5-1.5mm, large >1.5mm
        if length < 0.5:
            color = (0, 255, 0)  # Small - green
        elif length < 1.5:
            color = (255, 255, 0)  # Medium - yellow
        else:
            color = (255, 0, 0)  # Large - red
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label with measurements
        label = f"{length:.2f}×{width:.2f}mm"
        draw.text((x1, y1-35), label, fill=color, font=font_small)
    
    # Save overview
    image_name = image_path.stem
    overview_path = output_dir / f"{image_name}_overview.png"
    
    # Downsample if too large
    max_dim = 8000
    if img_draw.width > max_dim or img_draw.height > max_dim:
        scale = max_dim / max(img_draw.width, img_draw.height)
        new_size = (int(img_draw.width * scale), int(img_draw.height * scale))
        img_draw = img_draw.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  Downsampled to: {new_size}")
    
    img_draw.save(overview_path, quality=90)
    print(f"✓ Saved: {overview_path} ({img_draw.width}×{img_draw.height})")


def create_samples(img, data, image_path, output_dir, num_samples):
    """Create detailed view of sample organisms."""
    print(f"\nCreating samples visualization ({num_samples} organisms)...")
    
    # Sample organisms (mix of sizes)
    if len(data) > num_samples:
        # Stratified sample: mix of small, medium, large (in mm)
        small = data[data['length_mm'] < 0.5]
        medium = data[(data['length_mm'] >= 0.5) & (data['length_mm'] < 1.5)]
        large = data[data['length_mm'] >= 1.5]
        
        n_small = min(len(small), num_samples // 3)
        n_medium = min(len(medium), num_samples // 3)
        n_large = min(len(large), num_samples - n_small - n_medium)
        
        sample_data = pd.concat([
            small.sample(n=n_small) if len(small) > 0 else pd.DataFrame(),
            medium.sample(n=n_medium) if len(medium) > 0 else pd.DataFrame(),
            large.sample(n=n_large) if len(large) > 0 else pd.DataFrame()
        ])
    else:
        sample_data = data
    
    # Create grid
    n_cols = 5
    n_rows = (len(sample_data) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (_, row) in enumerate(sample_data.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract organism crop
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        
        # Add margin
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.width, x2 + margin)
        y2 = min(img.height, y2 + margin)
        
        crop = img.crop((x1, y1, x2, y2))
        crop_np = np.array(crop)
        
        # Display crop
        ax.imshow(crop_np)
        ax.axis('off')
        
        # Title with measurements
        title = f"ID {row['detection_id']}\n"
        title += f"{row['length_mm']:.2f} × {row['width_mm']:.2f} mm\n"
        title += f"Area: {row['area_mm2']:.4f} mm²\n"
        title += f"Vol: {row['volume_mm3']:.6f} mm³\n"
        title += f"Conf: {row['confidence_det']:.2f}"  # Use detection confidence
        
        ax.set_title(title, fontsize=9, fontfamily='monospace')
    
    # Hide unused axes
    for idx in range(len(sample_data), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    image_name = image_path.stem
    samples_path = output_dir / f"{image_name}_samples.png"
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {samples_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize organism measurements')
    parser.add_argument('--image', type=str, required=True, help='Original image')
    parser.add_argument('--detections', type=str, required=True, help='Detections CSV')
    parser.add_argument('--measurements', type=str, required=True, help='Measurements CSV')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--samples', type=int, default=50, help='Number of sample organisms')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    detections_csv = Path(args.detections)
    measurements_csv = Path(args.measurements)
    output_dir = Path(args.output)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    if not detections_csv.exists():
        print(f"Error: Detections not found: {detections_csv}")
        sys.exit(1)
    if not measurements_csv.exists():
        print(f"Error: Measurements not found: {measurements_csv}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_measurement_visualization(
        image_path, detections_csv, measurements_csv, 
        output_dir, args.samples
    )


if __name__ == '__main__':
    main()
