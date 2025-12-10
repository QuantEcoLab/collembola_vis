"""
Mine hard negatives from current detection false positives.

This script:
1. Loads current detection results (CSV files)
2. Matches detections to ground truth annotations (IoU >= 0.5)
3. Identifies false positives (no GT match)
4. Filters for "hard" negatives:
   - High classifier confidence (p >= 0.97)
   - Elongated shape (eccentricity >= 0.85)
   - Organism-like size (area 500-15000 px)
5. Saves hard negative crops for retraining
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Tuple
import shutil

from collembola_pipeline.config import DATA_DIR, CROPS_DIR, ANNOTATIONS_DIR, CSV_DIR, PLATES_DIR


def bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bboxes (x, y, w, h)"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to (x1, y1, x2, y2)
    box1 = (x1, y1, x1 + w1, y1 + h1)
    box2 = (x2, y2, x2 + w2, y2 + h2)
    
    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_detections_to_gt(
    detections_csv: Path,
    gt_csv: Path,
    plate_name_base: str,
    iou_thresh: float = 0.5
) -> pd.DataFrame:
    """Match detections to GT, return DataFrame with 'matched' column"""
    
    # Load detections
    df_det = pd.read_csv(detections_csv)
    
    # Load ground truth
    df_gt = pd.read_csv(gt_csv)
    
    # Filter GT for this plate
    df_gt_plate = df_gt[df_gt['id_collembole'].str.startswith(plate_name_base)].copy()
    
    # Match each detection
    matched = []
    for _, det in df_det.iterrows():
        det_bbox = (det['bbox_x'], det['bbox_y'], det['bbox_w'], det['bbox_h'])
        
        is_matched = False
        for _, gt in df_gt_plate.iterrows():
            gt_bbox = (gt['x'], gt['y'], gt['w'], gt['h'])
            iou = bbox_iou(det_bbox, gt_bbox)
            if iou >= iou_thresh:
                is_matched = True
                break
        
        matched.append(is_matched)
    
    df_det['matched'] = matched
    return df_det


def mine_hard_negatives(
    plate_path: Path,
    detections_csv: Path,
    gt_csv: Path,
    output_dir: Path,
    min_confidence: float = 0.97,
    min_eccentricity: float = 0.85,
    min_area: int = 500,
    max_area: int = 15000
) -> List[Path]:
    """
    Mine hard negative crops from false positive detections.
    
    Returns list of saved crop paths.
    """
    
    # Get plate name
    plate_name = plate_path.stem
    plate_name_base = plate_name.replace(' (1)', '')
    
    print(f"\n{'='*60}")
    print(f"Processing plate: {plate_name}")
    print(f"{'='*60}")
    
    # Match detections to GT
    df = match_detections_to_gt(detections_csv, gt_csv, plate_name_base)
    
    # Find false positives
    df_fp = df[~df['matched']].copy()
    print(f"Total detections: {len(df)}")
    print(f"True positives: {(df['matched']).sum()}")
    print(f"False positives: {len(df_fp)}")
    
    # Filter for hard negatives
    df_hard = df_fp[
        (df_fp['p_collembola'] >= min_confidence) &
        (df_fp['eccentricity'] >= min_eccentricity) &
        (df_fp['area_px'] >= min_area) &
        (df_fp['area_px'] <= max_area)
    ].copy()
    
    print(f"\nHard negatives (conf>={min_confidence}, ecc>={min_eccentricity}):")
    print(f"  Count: {len(df_hard)}")
    if len(df_hard) > 0:
        print(f"  Confidence range: {df_hard['p_collembola'].min():.4f} - {df_hard['p_collembola'].max():.4f}")
        print(f"  Eccentricity range: {df_hard['eccentricity'].min():.3f} - {df_hard['eccentricity'].max():.3f}")
        print(f"  Area range: {df_hard['area_px'].min():.0f} - {df_hard['area_px'].max():.0f} px")
    
    # Load plate image
    img = Image.open(plate_path)
    
    # Extract and save crops
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for idx, row in df_hard.iterrows():
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
        
        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.width, x + w + pad)
        y2 = min(img.height, y + h + pad)
        
        # Crop
        crop = img.crop((x1, y1, x2, y2))
        
        # Save
        crop_name = f"{plate_name_base}_hard_neg_{row['organism_id']:04d}.jpg"
        crop_path = output_dir / crop_name
        crop.save(crop_path, quality=95)
        saved_paths.append(crop_path)
    
    print(f"Saved {len(saved_paths)} hard negative crops to {output_dir}")
    
    return saved_paths


def update_crops_dataset_csv(
    crops_csv_path: Path,
    hard_neg_paths: List[Path]
) -> None:
    """Add hard negatives to crops_dataset.csv"""
    
    # Load existing
    df = pd.read_csv(crops_csv_path)
    
    # Create new rows
    new_rows = []
    for crop_path in hard_neg_paths:
        # Make path relative to project root
        if crop_path.is_absolute():
            try:
                rel_path = str(crop_path.relative_to(Path.cwd()))
            except ValueError:
                rel_path = str(crop_path)
        else:
            rel_path = str(crop_path)
        
        new_rows.append({
            'crop_id': rel_path,
            'collembola': False,
            'rel_x': 0.0,
            'rel_y': 0.0,
            'collembola_centers_abs': '[]',
            'collembola_centers_rel': '[]'
        })
    
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    
    # Save
    backup_path = crops_csv_path.parent / f"{crops_csv_path.stem}_backup.csv"
    shutil.copy(crops_csv_path, backup_path)
    print(f"\nBacked up original to: {backup_path}")
    
    df_combined.to_csv(crops_csv_path, index=False)
    print(f"Updated {crops_csv_path}")
    print(f"  Original rows: {len(df)}")
    print(f"  New hard negatives: {len(df_new)}")
    print(f"  Total rows: {len(df_combined)}")


def main():
    # Define plates to process
    plates = [
        {
            'plate': PLATES_DIR / "K1_Fe2O3001 (1).jpg",
            'csv': CSV_DIR / "K1_Fe2O3001 (1)_organisms.csv",
            'base_name': 'K1_Fe2O3001'
        },
        {
            'plate': PLATES_DIR / "C1_1_Fe2O3002 (1).jpg",
            'csv': CSV_DIR / "C1_1_Fe2O3002 (1)_organisms.csv",
            'base_name': 'C1_1_Fe2O3002'
        },
        {
            'plate': PLATES_DIR / "C5_2_Fe2O3003 (1).jpg",
            'csv': CSV_DIR / "C5_2_Fe2O3003 (1)_organisms.csv",
            'base_name': 'C5_2_Fe2O3003'
        }
    ]
    
    gt_csv = ANNOTATIONS_DIR / "collembolas_table.csv"
    output_dir = CROPS_DIR / "hard_negatives"
    
    all_hard_negatives = []
    
    for plate_info in plates:
        if not plate_info['csv'].exists():
            print(f"Skipping {plate_info['base_name']}: CSV not found")
            continue
        
        hard_negs = mine_hard_negatives(
            plate_path=plate_info['plate'],
            detections_csv=plate_info['csv'],
            gt_csv=gt_csv,
            output_dir=output_dir,
            min_confidence=0.97,
            min_eccentricity=0.85,
            min_area=500,
            max_area=15000
        )
        all_hard_negatives.extend(hard_negs)
    
    # Update crops_dataset.csv
    if all_hard_negatives:
        crops_csv = ANNOTATIONS_DIR / "crops_dataset.csv"
        update_crops_dataset_csv(crops_csv, all_hard_negatives)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: Mined {len(all_hard_negatives)} hard negatives total")
        print(f"{'='*60}")
    else:
        print("\nNo hard negatives found!")


if __name__ == '__main__':
    main()
