#!/usr/bin/env python3
"""
Analyze ruler inspection images to understand ruler characteristics.

This script analyzes the corner images produced by inspect_ruler.py
to help identify:
1. Which corner has the ruler
2. What the ruler looks like (color, size, orientation)
3. Optimal detection parameters

Usage:
    python scripts/analyze_ruler.py --input ruler_inspection/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def analyze_corner(corner_name: str, image_dir: Path):
    """
    Analyze a single corner to check for ruler presence.
    """
    # Load images
    orig_path = image_dir / f"{corner_name}_original.jpg"
    if not orig_path.exists():
        print(f"⚠ {corner_name}: Image not found")
        return None
    
    img = cv2.imread(str(orig_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    
    # Check edges (rulers often on image borders)
    border = 100
    
    # Analyze each edge
    edges = {
        'top': gray[:border, :],
        'bottom': gray[-border:, :],
        'left': gray[:, :border],
        'right': gray[:, -border:],
    }
    
    results = {}
    
    for edge_name, edge_region in edges.items():
        # Calculate statistics
        mean_val = edge_region.mean()
        std_val = edge_region.std()
        
        # Edge detection in border region
        edge_img = cv2.Canny(edge_region, 50, 150)
        edge_density = edge_img.sum() / edge_img.size
        
        # Look for horizontal/vertical structures
        if edge_name in ['top', 'bottom']:
            # Check for horizontal lines (ruler tick marks)
            profile = edge_region.mean(axis=0)
        else:
            # Check for vertical lines
            profile = edge_region.mean(axis=1)
        
        # Detect periodic structures (ruler tick marks)
        profile_std = np.std(profile)
        
        results[edge_name] = {
            'mean': mean_val,
            'std': std_val,
            'edge_density': edge_density,
            'profile_std': profile_std,
        }
    
    # Score each edge (rulers have high contrast, periodic patterns)
    scores = {}
    for edge_name, stats in results.items():
        # Rulers typically have:
        # - High contrast (high std)
        # - Periodic markings (high profile std)
        # - Edges for tick marks (high edge density)
        score = stats['edge_density'] * 1000 + stats['profile_std'] * 0.1
        scores[edge_name] = score
    
    # Find best edge
    best_edge = max(scores, key=scores.get)
    best_score = scores[best_edge]
    
    # Check if score suggests ruler presence
    has_ruler = best_score > 10  # Threshold to be tuned
    
    return {
        'corner': corner_name,
        'has_ruler': has_ruler,
        'best_edge': best_edge,
        'score': best_score,
        'edge_stats': results,
        'all_scores': scores,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ruler inspection images',
    )
    
    parser.add_argument('--input', type=str, default='ruler_inspection',
                        help='Directory with inspection images (default: ruler_inspection)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        print("Run inspect_ruler.py first to generate inspection images")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"RULER ANALYSIS")
    print(f"{'='*70}")
    
    corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    results = []
    
    for corner in corners:
        result = analyze_corner(corner, input_dir)
        if result:
            results.append(result)
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nRULER DETECTION SCORES:")
    print(f"{'Corner':<15} {'Best Edge':<12} {'Score':<10} {'Ruler?'}")
    print(f"{'-'*50}")
    
    for r in results:
        ruler_str = "✓ YES" if r['has_ruler'] else "  no"
        print(f"{r['corner']:<15} {r['best_edge']:<12} {r['score']:<10.2f} {ruler_str}")
    
    print(f"\nDETAILED ANALYSIS:")
    print(f"{'-'*70}")
    
    for r in results[:2]:  # Show top 2
        print(f"\n{r['corner'].upper()}:")
        print(f"  Most likely ruler location: {r['best_edge']} edge")
        print(f"  Edge scores:")
        for edge, score in r['all_scores'].items():
            marker = " ← highest" if edge == r['best_edge'] else ""
            print(f"    {edge:<8}: {score:>8.2f}{marker}")
        
        print(f"  Edge statistics:")
        for edge, stats in r['edge_stats'].items():
            if edge == r['best_edge']:
                print(f"    {edge} (BEST):")
                print(f"      Edge density: {stats['edge_density']:.4f}")
                print(f"      Profile std:  {stats['profile_std']:.2f}")
    
    print(f"\n{'='*70}")
    
    if results and results[0]['has_ruler']:
        best = results[0]
        print(f"RECOMMENDATION:")
        print(f"  Ruler most likely in: {best['corner']} corner, {best['best_edge']} edge")
        print(f"  Score: {best['score']:.2f}")
        print(f"\nNEXT STEP:")
        print(f"  Examine: {input_dir}/{best['corner']}_original.jpg")
        print(f"  Look at the {best['best_edge']} edge to confirm ruler presence")
    else:
        print(f"⚠ No strong ruler signal detected")
        print(f"  Check inspection images manually: {input_dir}/")
        print(f"  Ruler may be smaller or have different appearance")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
