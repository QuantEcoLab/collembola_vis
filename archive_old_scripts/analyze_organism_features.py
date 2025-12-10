#!/usr/bin/env python3
"""
Analyze organism appearance to design better CV detection.

This script samples GT organisms and analyzes their visual characteristics
to understand what features make them detectable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def analyze_organisms(image_path: Path, num_samples: int = 30):
    """Analyze visual characteristics of GT organisms."""
    
    # Load image and GT
    img = np.array(Image.open(image_path).convert('RGB'))
    H, W = img.shape[:2]
    
    gt_csv = Path('data/annotations/collembolas_table.csv')
    gt_df = pd.read_csv(gt_csv)
    plate_prefix = image_path.stem.split('_')[0]
    gt_df = gt_df[gt_df['id_collembole'].str.contains(plate_prefix)]
    
    print(f"Image: {image_path.name}")
    print(f"Size: {W}x{H}")
    print(f"Ground truth organisms: {len(gt_df)}")
    print()
    
    # Sample organisms
    samples = gt_df.sample(min(num_samples, len(gt_df)), random_state=42)
    
    # Create visualization grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    # Analyze each organism
    stats = {
        'contrast': [],
        'brightness': [],
        'texture_std': [],
        'edges': [],
        'area': []
    }
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        
        # Extract crop with padding
        pad = 30
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(W, x+w+pad), min(H, y+h+pad)
        crop = img[y0:y1, x0:x1]
        
        # Get organism and background regions
        org_y0, org_y1 = y - y0, y - y0 + h
        org_x0, org_x1 = x - x0, x - x0 + w
        org_pixels = crop[org_y0:org_y1, org_x0:org_x1]
        
        # Background: border pixels
        bg_mask = np.ones(crop.shape[:2], dtype=bool)
        bg_mask[org_y0:org_y1, org_x0:org_x1] = False
        bg_pixels = crop[bg_mask]
        
        # Calculate features
        org_gray = cv2.cvtColor(org_pixels, cv2.COLOR_RGB2GRAY)
        bg_gray = cv2.cvtColor(bg_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # Contrast
        contrast = abs(org_gray.mean() - bg_gray.mean())
        
        # Texture
        texture_std = org_gray.std()
        
        # Edges (Sobel)
        sobelx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_in_org = edge_magnitude[org_y0:org_y1, org_x0:org_x1].mean()
        
        stats['contrast'].append(contrast)
        stats['brightness'].append(org_gray.mean())
        stats['texture_std'].append(texture_std)
        stats['edges'].append(edge_in_org)
        stats['area'].append(w * h)
        
        # Visualize
        ax = axes[idx]
        
        # Draw bbox on crop
        vis_crop = crop.copy()
        cv2.rectangle(vis_crop, (org_x0, org_y0), (org_x1, org_y1), (255, 0, 0), 2)
        
        ax.imshow(vis_crop)
        ax.set_title(f'#{idx+1}\nC={contrast:.1f} T={texture_std:.1f}\nE={edge_in_org:.1f} A={w*h}', 
                     fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Organism Samples from {image_path.name}', fontsize=16)
    plt.tight_layout()
    
    output_path = Path('outputs/organism_samples_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Print statistics
    print(f"\nOrganism Characteristics (n={len(samples)}):")
    print(f"  Contrast: {np.mean(stats['contrast']):.1f} ± {np.std(stats['contrast']):.1f} (range: {np.min(stats['contrast']):.1f}-{np.max(stats['contrast']):.1f})")
    print(f"  Brightness: {np.mean(stats['brightness']):.1f} ± {np.std(stats['brightness']):.1f}")
    print(f"  Texture std: {np.mean(stats['texture_std']):.1f} ± {np.std(stats['texture_std']):.1f}")
    print(f"  Edge magnitude: {np.mean(stats['edges']):.1f} ± {np.std(stats['edges']):.1f}")
    print(f"  Area: {np.mean(stats['area']):.0f} ± {np.std(stats['area']):.0f} px²")
    
    # Key insights
    print("\nKey Insights:")
    low_contrast_pct = 100 * sum(1 for c in stats['contrast'] if c < 15) / len(stats['contrast'])
    print(f"  - {low_contrast_pct:.0f}% of organisms have contrast < 15 (very low!)")
    
    high_texture_pct = 100 * sum(1 for t in stats['texture_std'] if t > 20) / len(stats['texture_std'])
    print(f"  - {high_texture_pct:.0f}% of organisms have texture std > 20 (textured)")
    
    high_edge_pct = 100 * sum(1 for e in stats['edges'] if e > 30) / len(stats['edges'])
    print(f"  - {high_edge_pct:.0f}% of organisms have edge magnitude > 30")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/slike/K1_Fe2O3001 (1).jpg"))
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    
    stats = analyze_organisms(args.image, args.samples)
