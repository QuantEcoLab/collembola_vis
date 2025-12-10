#!/usr/bin/env python3
"""
Visualize region proposals (CV or SAM) before classification.

This helps diagnose whether proposals are accurate or if the problem
is in the proposal stage vs classification stage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collembola_pipeline.proposal_cv_fast import propose_regions_cv_fast
from collembola_pipeline.proposal_sam import propose_regions_sam_fast


def visualize_proposals(
    image_path: Path,
    method: str = 'cv',
    show_ground_truth: bool = True,
    max_display: int = 500,
    output_path: Path = None,
    figsize: tuple = (20, 20)
):
    """
    Visualize region proposals overlaid on the image.
    
    Args:
        image_path: Path to plate image
        method: 'cv' or 'sam'
        show_ground_truth: If True, also show GT bboxes
        max_display: Maximum number of proposals to display
        output_path: Path to save visualization (if None, show interactively)
        figsize: Figure size in inches
    """
    print(f"\n{'='*60}")
    print(f"Visualizing {method.upper()} proposals: {image_path.name}")
    print(f"{'='*60}\n")
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    H, W = image.shape[:2]
    
    print(f"Image size: {W}x{H}")
    
    # Generate proposals
    print(f"\nGenerating {method.upper()} proposals...")
    
    if method == 'cv':
        proposals, plate_circle = propose_regions_cv_fast(
            image,
            bbox_scale_factor=4.5,
            min_area=1000,
            max_area=100000,
            min_eccentricity=0.60,
            detect_plate=True,
            plate_shrink=50,
            verbose=True
        )
    else:  # sam
        proposals = propose_regions_sam_fast(
            image,
            device='cuda',
            min_area=1000,
            max_area=100000,
            verbose=True
        )
        plate_circle = None
    
    print(f"\nFound {len(proposals)} proposals")
    
    # Load ground truth if requested
    gt_bboxes = []
    if show_ground_truth:
        gt_csv = Path('data/annotations/collembolas_table.csv')
        if gt_csv.exists():
            gt_df = pd.read_csv(gt_csv)
            # Filter for this plate
            plate_prefix = image_path.stem.split('_')[0]
            gt_df = gt_df[gt_df['id_collembole'].str.contains(plate_prefix)]
            gt_bboxes = [(row['x'], row['y'], row['w'], row['h']) for _, row in gt_df.iterrows()]
            print(f"Loaded {len(gt_bboxes)} ground truth bboxes")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Downsample image for display (10k x 10k is too large)
    display_scale = min(1.0, 3000 / max(H, W))
    if display_scale < 1.0:
        display_h = int(H * display_scale)
        display_w = int(W * display_scale)
        display_image = np.array(Image.fromarray(image).resize((display_w, display_h)))
        print(f"Downsampling image for display: {W}x{H} -> {display_w}x{display_h}")
    else:
        display_image = image
        display_scale = 1.0
    
    ax.imshow(display_image)
    ax.set_title(f'{method.upper()} Proposals: {len(proposals)} regions', fontsize=16)
    ax.axis('off')
    
    # Draw plate circle if detected
    if plate_circle:
        cx, cy, r = plate_circle
        circle_patch = plt.Circle(
            (cx * display_scale, cy * display_scale),
            r * display_scale,
            color='cyan',
            fill=False,
            linewidth=2,
            linestyle='--',
            label='Detected Plate'
        )
        ax.add_patch(circle_patch)
    
    # Draw ground truth bboxes (green)
    if gt_bboxes:
        for x, y, w, h in gt_bboxes[:max_display]:
            rect = patches.Rectangle(
                (x * display_scale, y * display_scale),
                w * display_scale,
                h * display_scale,
                linewidth=1,
                edgecolor='lime',
                facecolor='none',
                alpha=0.6
            )
            ax.add_patch(rect)
    
    # Draw proposals (red)
    num_to_show = min(len(proposals), max_display)
    for i, prop in enumerate(proposals[:num_to_show]):
        x, y, w, h = prop.bbox
        rect = patches.Rectangle(
            (x * display_scale, y * display_scale),
            w * display_scale,
            h * display_scale,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none',
            alpha=0.4
        )
        ax.add_patch(rect)
    
    if len(proposals) > max_display:
        print(f"Warning: Only showing first {max_display}/{len(proposals)} proposals")
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label=f'{method.upper()} Proposals ({len(proposals)})'),
    ]
    if gt_bboxes:
        legend_elements.append(
            Line2D([0], [0], color='lime', linewidth=2, label=f'Ground Truth ({len(gt_bboxes)})')
        )
    if plate_circle:
        legend_elements.append(
            Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='Plate Boundary')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
    else:
        plt.show()
    
    # Calculate recall (how many GT bboxes have overlapping proposals)
    if gt_bboxes and proposals:
        matched_gt = 0
        for gt_x, gt_y, gt_w, gt_h in gt_bboxes:
            gt_cx = gt_x + gt_w / 2
            gt_cy = gt_y + gt_h / 2
            
            # Check if any proposal contains GT center
            for prop in proposals:
                px, py, pw, ph = prop.bbox
                if px <= gt_cx <= px + pw and py <= gt_cy <= py + ph:
                    matched_gt += 1
                    break
        
        recall = 100 * matched_gt / len(gt_bboxes) if gt_bboxes else 0
        print(f"\nProposal Recall: {matched_gt}/{len(gt_bboxes)} ({recall:.1f}%)")
        print(f"  (How many GT organisms are covered by at least one proposal)")
    
    # Proposal statistics
    if proposals:
        areas = [p.area for p in proposals]
        confidences = [p.confidence for p in proposals]
        
        print(f"\nProposal Statistics:")
        print(f"  Area: {min(areas)}-{max(areas)} px (median: {np.median(areas):.0f})")
        print(f"  Confidence: {min(confidences):.3f}-{max(confidences):.3f} (median: {np.median(confidences):.3f})")
        
        # Check edge proximity
        edge_margin = 200
        edge_count = sum(
            1 for p in proposals
            if p.bbox[0] < edge_margin or p.bbox[1] < edge_margin
            or p.bbox[0] + p.bbox[2] > W - edge_margin
            or p.bbox[1] + p.bbox[3] > H - edge_margin
        )
        print(f"  Near edges (<{edge_margin}px): {edge_count}/{len(proposals)} ({100*edge_count/len(proposals):.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize region proposals")
    parser.add_argument("image", type=Path, help="Path to plate image")
    parser.add_argument("--method", choices=['cv', 'sam'], default='cv', help="Proposal method")
    parser.add_argument("--no-gt", dest='show_gt', action='store_false', help="Don't show ground truth")
    parser.add_argument("--max-display", type=int, default=500, help="Max proposals to display")
    parser.add_argument("--output", type=Path, help="Output path for visualization")
    parser.add_argument("--figsize", type=int, nargs=2, default=[20, 20], help="Figure size (width height)")
    
    args = parser.parse_args()
    
    visualize_proposals(
        args.image,
        method=args.method,
        show_ground_truth=args.show_gt,
        max_display=args.max_display,
        output_path=args.output,
        figsize=tuple(args.figsize)
    )
