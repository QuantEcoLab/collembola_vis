"""
Debug CV proposal issues by comparing with ground truth.

This script:
1. Loads K1 ground truth bboxes
2. Generates CV proposals
3. Compares distributions and spatial locations
4. Identifies edge artifacts and other issues
"""

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from collembola_pipeline.proposal_cv_fast import propose_regions_cv_fast


def load_ground_truth(plate_name: str) -> pd.DataFrame:
    """Load ground truth annotations for a plate."""
    csv_path = Path("data/annotations/collembolas_table.csv")
    df = pd.read_csv(csv_path)
    # Filter for this plate (remove extension from plate_name for matching)
    plate_stem = Path(plate_name).stem
    df = df[df['id_collembole'].str.contains(plate_stem.split('_')[0])]
    return df


def analyze_proposals(image_path: Path, verbose: bool = True):
    """Analyze CV proposals vs ground truth."""
    
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Image: {image_path.name}")
        print(f"Shape: {image.shape}")
        print(f"{'='*60}\n")
    
    # Load ground truth
    gt_df = load_ground_truth(image_path.name)
    if verbose:
        print(f"Ground Truth: {len(gt_df)} organisms\n")
        print("GT Bbox Statistics:")
        print(f"  x: {gt_df['x'].min()}-{gt_df['x'].max()} (median: {gt_df['x'].median():.0f})")
        print(f"  y: {gt_df['y'].min()}-{gt_df['y'].max()} (median: {gt_df['y'].median():.0f})")
        print(f"  w: {gt_df['w'].min()}-{gt_df['w'].max()} (median: {gt_df['w'].median():.0f})")
        print(f"  h: {gt_df['h'].min()}-{gt_df['h'].max()} (median: {gt_df['h'].median():.0f})")
        gt_areas = gt_df['w'] * gt_df['h']
        print(f"  area: {gt_areas.min()}-{gt_areas.max()} (median: {gt_areas.median():.0f})")
        print()
    
    # Generate CV proposals with different parameters
    print("Testing CV proposals with different parameters...\n")
    
    test_configs = [
        {"name": "Original", "bbox_scale_factor": 3.0, "min_area": 1000, "max_area": 100000},
        {"name": "Larger scale", "bbox_scale_factor": 4.5, "min_area": 1000, "max_area": 100000},
        {"name": "Wider area range", "bbox_scale_factor": 4.5, "min_area": 500, "max_area": 150000},
        {"name": "No eccentricity", "bbox_scale_factor": 4.5, "min_area": 1000, "max_area": 100000, "min_eccentricity": 0.0},
    ]
    
    results = {}
    
    for config in test_configs:
        name = config.pop("name")
        print(f"\n{name}: {config}")
        proposals = propose_regions_cv_fast(image, verbose=False, **config)
        
        if len(proposals) > 0:
            # Analyze proposal distribution
            prop_areas = [p.area for p in proposals]
            prop_xs = [p.bbox[0] for p in proposals]
            prop_ys = [p.bbox[1] for p in proposals]
            prop_ws = [p.bbox[2] for p in proposals]
            prop_hs = [p.bbox[3] for p in proposals]
            
            print(f"  Proposals: {len(proposals)}")
            print(f"  x: {min(prop_xs)}-{max(prop_xs)} (median: {np.median(prop_xs):.0f})")
            print(f"  y: {min(prop_ys)}-{max(prop_ys)} (median: {np.median(prop_ys):.0f})")
            print(f"  w: {min(prop_ws)}-{max(prop_ws)} (median: {np.median(prop_ws):.0f})")
            print(f"  h: {min(prop_hs)}-{max(prop_hs)} (median: {np.median(prop_hs):.0f})")
            print(f"  area: {min(prop_areas)}-{max(prop_areas)} (median: {np.median(prop_areas):.0f})")
            
            # Check edge proximity (within 200px of edges)
            edge_margin = 200
            edge_proposals = sum(
                1 for p in proposals
                if p.bbox[0] < edge_margin or p.bbox[1] < edge_margin
                or p.bbox[0] + p.bbox[2] > W - edge_margin
                or p.bbox[1] + p.bbox[3] > H - edge_margin
            )
            print(f"  Near edges (<{edge_margin}px): {edge_proposals}/{len(proposals)} ({100*edge_proposals/len(proposals):.1f}%)")
            
            # Calculate recall (how many GT organisms have overlapping proposals)
            matched_gt = 0
            for _, gt_row in gt_df.iterrows():
                gt_x, gt_y, gt_w, gt_h = gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']
                gt_cx, gt_cy = gt_x + gt_w/2, gt_y + gt_h/2
                
                # Check if any proposal contains GT center
                for prop in proposals:
                    px, py, pw, ph = prop.bbox
                    if px <= gt_cx <= px + pw and py <= gt_cy <= py + ph:
                        matched_gt += 1
                        break
            
            recall = matched_gt / len(gt_df) if len(gt_df) > 0 else 0.0
            print(f"  Recall (GT coverage): {matched_gt}/{len(gt_df)} ({100*recall:.1f}%)")
            
            results[name] = {
                "proposals": proposals,
                "count": len(proposals),
                "recall": recall,
                "edge_count": edge_proposals
            }
        else:
            print(f"  No proposals generated!")
            results[name] = {"proposals": [], "count": 0, "recall": 0.0, "edge_count": 0}
    
    # Find best config
    best_config = max(results.items(), key=lambda x: x[1]['recall'])
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_config[0]}")
    print(f"  Recall: {100*best_config[1]['recall']:.1f}%")
    print(f"  Proposals: {best_config[1]['count']}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/slike/K1_Fe2O3001 (1).jpg"))
    args = parser.parse_args()
    
    results = analyze_proposals(args.image)
