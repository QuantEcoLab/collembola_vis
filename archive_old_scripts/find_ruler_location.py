#!/usr/bin/env python3
"""
Search for ruler in the top half, center region of image.

Creates a visualization showing where the ruler might be located.

Usage:
    python scripts/find_ruler_location.py --image "data/slike/K1_Fe2O3001 (1).jpg"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_ruler_location(image_path: Path, output_path: Path = Path('ruler_location_search.png')):
    """
    Search for ruler in top half, center region.
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Extract TOP HALF of image
    top_half_height = img.height // 2
    print(f"\nExtracting top half (height={top_half_height}px)...")
    top_half = img.crop((0, 0, img.width, top_half_height))
    top_np = np.array(top_half.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(top_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Create a grid visualization - divide into regions
    print(f"Analyzing image in grid...")
    
    # Define center region (middle third horizontally)
    center_left = w // 3
    center_right = 2 * w // 3
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('RULER LOCATION SEARCH - Top Half of Image\n(Ruler should be in CENTER horizontally)', 
                 fontsize=16, fontweight='bold')
    
    # Define 9 regions to check
    regions = [
        ('Top-Left', 0, w//3, 0, h//3),
        ('Top-Center', w//3, 2*w//3, 0, h//3),
        ('Top-Right', 2*w//3, w, 0, h//3),
        ('Mid-Left', 0, w//3, h//3, 2*h//3),
        ('Mid-Center', w//3, 2*w//3, h//3, 2*h//3),
        ('Mid-Right', 2*w//3, w, h//3, 2*h//3),
        ('Bottom-Left', 0, w//3, 2*h//3, h),
        ('Bottom-Center', w//3, 2*w//3, 2*h//3, h),
        ('Bottom-Right', 2*w//3, w, 2*h//3, h),
    ]
    
    scores = []
    
    for idx, (name, x1, x2, y1, y2) in enumerate(regions):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Extract region
        region = gray[y1:y2, x1:x2]
        
        # Analyze for ruler-like features
        # Rulers have strong horizontal lines and periodic vertical marks
        edges = cv2.Canny(region, 50, 150)
        
        # Count horizontal edges (ruler body)
        horizontal_edges = np.sum(edges, axis=1).mean()
        
        # Count vertical edges (tick marks)
        vertical_edges = np.sum(edges, axis=0).mean()
        
        # Ruler score: high horizontal + periodic vertical
        score = horizontal_edges * 0.3 + vertical_edges * 0.7
        scores.append((name, score, x1, x2, y1, y2))
        
        # Show region
        ax.imshow(region, cmap='gray')
        ax.set_title(f'{name}\nScore: {score:.1f}', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'x: {x1}-{x2}')
        ax.set_ylabel(f'y: {y1}-{y2}')
        
        # Highlight if likely ruler location (center regions)
        if 'Center' in name:
            for spine in ax.spines.values():
                spine.set_edgecolor('lime')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved search visualization to: {output_path}")
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"RULER LOCATION ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Region':<15} {'Score':<10} {'X Range':<15} {'Y Range':<15}")
    print(f"{'-'*70}")
    for name, score, x1, x2, y1, y2 in scores[:5]:
        print(f"{name:<15} {score:<10.1f} {x1}-{x2:<10} {y1}-{y2}")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION:")
    best_name, best_score, bx1, bx2, by1, by2 = scores[0]
    print(f"  Highest score region: {best_name}")
    print(f"  Score: {best_score:.1f}")
    print(f"  Position: x={bx1}-{bx2}, y={by1}-{by2}")
    print(f"\nNEXT STEP:")
    print(f"  Open {output_path} to visually confirm ruler location")
    print(f"  Look for the ruler in the highlighted CENTER regions")
    print(f"{'='*70}")
    
    plt.close()
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Search for ruler location in top half of image',
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image')
    parser.add_argument('--output', type=str, default='ruler_location_search.png',
                        help='Output visualization (default: ruler_location_search.png)')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    find_ruler_location(
        image_path=image_path,
        output_path=Path(args.output)
    )


if __name__ == '__main__':
    main()
