#!/usr/bin/env python3
"""
Test new CV-based detection pipeline on sample plates.

Usage:
    python scripts/04_process_plates.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collembola_pipeline.detect_organisms import detect_organisms, batch_process_plates

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "slike"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Test images
TEST_IMAGES = [
    IMAGES_DIR / "K1_Fe2O3001 (1).jpg",
    IMAGES_DIR / "C1_1_Fe2O3002 (1).jpg",
    IMAGES_DIR / "C5_2_Fe2O3003 (1).jpg",
]


def test_single_plate():
    """Test on a single plate (K1)"""
    print("\n" + "="*80)
    print("TESTING SINGLE PLATE: K1 (Fast CV Proposals)")
    print("="*80)
    
    detections_df, overlay = detect_organisms(
        TEST_IMAGES[0],
        output_dir=OUTPUT_DIR / "single_test",
        proposal_method='cv',  # Use fast dilated CV
        use_watershed=False,
        confidence_threshold=0.5,  # Lower threshold for better recall
        batch_size=64,
        device='cuda',
        verbose=True
    )
    
    print(f"\nDetections shape: {detections_df.shape}")
    if len(detections_df) > 0:
        print(f"\nFirst 5 detections:")
        print(detections_df.head().to_string())
        
        print(f"\nMorphology statistics:")
        print(detections_df[['length_um', 'width_um', 'volume_um3']].describe())


def test_batch_plates():
    """Test batch processing on all 3 plates"""
    print("\n" + "="*80)
    print("TESTING BATCH PROCESSING: K1, C1, C5 (Fast CV)")
    print("="*80)
    
    results = batch_process_plates(
        TEST_IMAGES,
        output_dir=OUTPUT_DIR / "batch_test",
        proposal_method='cv',  # Fast CV
        use_watershed=False,
        confidence_threshold=0.5,  # Lower threshold
        batch_size=64,
        device='cuda',
        verbose=True
    )
    
    return results


def compare_with_ground_truth():
    """Compare detections with ground truth annotations"""
    import pandas as pd
    
    print("\n" + "="*80)
    print("COMPARISON WITH GROUND TRUTH")
    print("="*80)
    
    # Load ground truth
    gt_path = DATA_DIR / "annotations" / "collembolas_table.csv"
    gt_df = pd.read_csv(gt_path)
    
    # Ground truth counts by plate
    gt_counts = gt_df.groupby('image_name').size().to_dict()
    
    # Map image names (remove " (1)" suffix)
    plate_mapping = {
        'K1_Fe2O3001 (1).jpg': 'K1_Fe2O3001.jpg',
        'C1_1_Fe2O3002 (1).jpg': 'C1_1_Fe2O3002.jpg',
        'C5_2_Fe2O3003 (1).jpg': 'C5_2_Fe2O3003.jpg',
    }
    
    print("\nGround Truth Counts:")
    for test_img, gt_img in plate_mapping.items():
        count = gt_counts.get(gt_img, 0)
        print(f"  {gt_img}: {count}")
    
    # Load our detections
    detection_counts = {}
    for img_path in TEST_IMAGES:
        csv_path = OUTPUT_DIR / "batch_test" / f"{img_path.stem}_organisms.csv"
        if csv_path.exists():
            det_df = pd.read_csv(csv_path)
            detection_counts[img_path.name] = len(det_df)
        else:
            detection_counts[img_path.name] = 0
    
    print("\nOur Detection Counts:")
    for img_name, count in detection_counts.items():
        print(f"  {img_name}: {count}")
    
    print("\nComparison:")
    print(f"{'Plate':<25} {'Ground Truth':<15} {'Detected':<15} {'Difference':<15}")
    print("-" * 70)
    
    for test_img, gt_img in plate_mapping.items():
        gt_count = gt_counts.get(gt_img, 0)
        det_count = detection_counts.get(test_img, 0)
        diff = det_count - gt_count
        diff_pct = (diff / gt_count * 100) if gt_count > 0 else 0
        
        print(f"{gt_img:<25} {gt_count:<15} {det_count:<15} {diff:+d} ({diff_pct:+.1f}%)")
    
    total_gt = sum(gt_counts.get(plate_mapping[img], 0) for img in detection_counts.keys())
    total_det = sum(detection_counts.values())
    total_diff = total_det - total_gt
    total_diff_pct = (total_diff / total_gt * 100) if total_gt > 0 else 0
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_gt:<15} {total_det:<15} {total_diff:+d} ({total_diff_pct:+.1f}%)")
    
    print("\nNote: Ground truth may be incomplete (we found many real organisms not annotated)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test collembola detection pipeline")
    parser.add_argument("--mode", choices=["single", "batch", "compare", "all"], 
                        default="all", help="Test mode")
    
    args = parser.parse_args()
    
    if args.mode in ["single", "all"]:
        test_single_plate()
    
    if args.mode in ["batch", "all"]:
        test_batch_plates()
    
    if args.mode in ["compare", "all"]:
        compare_with_ground_truth()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
