#!/usr/bin/env python3
"""
Convert ImageJ ROI annotations to YOLO format.

This script takes the CSV output from convert_imagej_rois.py and creates
a YOLO-compatible dataset with train/val/test splits.

YOLO format:
- Images in: images/train/, images/val/, images/test/
- Labels in: labels/train/, labels/val/, labels/test/
- Label format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
  where all coordinates are normalized to [0, 1]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import shutil
from PIL import Image
import yaml
import numpy as np
from typing import List


def split_plates(plates: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42):
    """
    Split plates into train/val/test sets.
    
    Args:
        plates: List of plate IDs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_plates, val_plates, test_plates)
    """
    np.random.seed(random_seed)
    plates = sorted(plates)  # Consistent ordering
    n_plates = len(plates)
    
    # Shuffle
    shuffled = np.random.permutation(plates)
    
    # Calculate split indices
    n_train = int(n_plates * train_ratio)
    n_val = int(n_plates * val_ratio)
    
    train_plates = shuffled[:n_train].tolist()
    val_plates = shuffled[n_train:n_train + n_val].tolist()
    test_plates = shuffled[n_train + n_val:].tolist()
    
    return train_plates, val_plates, test_plates


def prepare_yolo_from_imagej_rois(
    rois_csv: Path,
    images_base_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Convert ImageJ ROI CSV to YOLO format dataset.
    
    Args:
        rois_csv: Path to imagej_rois_bez.csv
        images_base_dir: Base directory containing plate images
        output_dir: Output directory for YOLO dataset
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        random_seed: Random seed for reproducibility
    """
    print("="*70)
    print("ImageJ ROIs to YOLO Format Converter")
    print("="*70)
    print()
    
    # Load ROI annotations
    df = pd.read_csv(rois_csv)
    print(f"Loaded {len(df):,} ROI annotations from {rois_csv}")
    print(f"Plates: {df['plate_id'].nunique()}")
    print()
    
    # Split plates into train/val/test
    plates = df['plate_id'].unique()
    train_plates, val_plates, test_plates = split_plates(
        plates, train_ratio, val_ratio, random_seed
    )
    
    print(f"Dataset split (random_seed={random_seed}):")
    print(f"  Train: {len(train_plates)} plates - {train_plates}")
    print(f"  Val:   {len(val_plates)} plates - {val_plates}")
    print(f"  Test:  {len(test_plates)} plates - {test_plates}")
    print()
    
    # Create output structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    stats = {'train': 0, 'val': 0, 'test': 0}
    split_mapping = {
        'train': train_plates,
        'val': val_plates,
        'test': test_plates
    }
    
    processed_images = set()
    
    for split, plates_in_split in split_mapping.items():
        for plate_id in plates_in_split:
            plate_annots = df[df['plate_id'] == plate_id]
            
            if len(plate_annots) == 0:
                continue
            
            # Get image name (assume all rows for same plate have same image_name)
            image_name = plate_annots.iloc[0]['image_name']
            
            # Find the actual image file
            # Try multiple locations
            image_path = None
            search_paths = [
                images_base_dir / "Maxima_ROI setovi" / "Fe2O3" / f"ROI_{plate_id}" / image_name,
                images_base_dir / "Luca_ROI setovi" / "Mikroplastika" / plate_id / image_name,
            ]
            
            for path in search_paths:
                if path.exists():
                    image_path = path
                    break
            
            if image_path is None or not image_path.exists():
                print(f"WARNING: Image not found for {plate_id}: {image_name}")
                print(f"  Searched: {[str(p) for p in search_paths]}")
                continue
            
            # Load image to get dimensions
            try:
                img = Image.open(image_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"ERROR loading image {image_path}: {e}")
                continue
            
            # Copy image
            dest_image = output_dir / "images" / split / image_name
            shutil.copy(image_path, dest_image)
            processed_images.add(image_name)
            
            # Convert annotations to YOLO format
            yolo_labels = []
            for _, row in plate_annots.iterrows():
                x, y, w, h = row['x'], row['y'], row['w'], row['h']
                
                # Convert to YOLO format (normalized center coords + normalized size)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Class 0 = collembola (only one class)
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Write labels file
            label_name = image_name.replace('.jpg', '.txt').replace('.JPG', '.txt')
            dest_label = output_dir / "labels" / split / label_name
            with open(dest_label, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            stats[split] += len(yolo_labels)
            print(f"{split.upper()}: {plate_id:30s} ({image_name:30s}) - {len(yolo_labels):4d} ROIs")
    
    print()
    print("="*70)
    print("Dataset Statistics:")
    print("="*70)
    total_rois = sum(stats.values())
    for split in ["train", "val", "test"]:
        count = stats[split]
        pct = 100 * count / total_rois if total_rois > 0 else 0
        n_plates = len(split_mapping[split])
        print(f"  {split.capitalize():5s}: {count:5d} ROIs ({pct:5.1f}%) across {n_plates:2d} plates")
    print(f"  Total:  {total_rois:5d} ROIs")
    print()
    
    # Create YOLO data.yaml configuration
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Number of classes
        'names': ['collembola']  # Class names
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✓ Created YOLO configuration: {yaml_path}")
    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/  ({stats['train']} ROIs, {len(train_plates)} plates)")
    print(f"    │   ├── val/    ({stats['val']} ROIs, {len(val_plates)} plates)")
    print(f"    │   └── test/   ({stats['test']} ROIs, {len(test_plates)} plates)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/")
    print(f"    │   ├── val/")
    print(f"    │   └── test/")
    print(f"    └── data.yaml")
    print()
    print("✓ YOLO dataset ready for training!")
    print()
    print("Next steps:")
    print(f"  1. Train YOLO model:")
    print(f"     yolo detect train data={output_dir}/data.yaml model=yolo11n.pt epochs=100 imgsz=640")
    print(f"  2. Validate:")
    print(f"     yolo detect val model=runs/detect/train/weights/best.pt data={output_dir}/data.yaml")
    print(f"  3. Predict on new images:")
    print(f"     yolo detect predict model=runs/detect/train/weights/best.pt source=data/slike/")
    print()
    
    return output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ImageJ ROI CSV to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert ImageJ ROIs to YOLO format with default 70/15/15 split
  python scripts/imagej_rois_to_yolo.py \\
      --rois-csv data/annotations/imagej_rois_bez.csv \\
      --images-dir "data/training_data/Collembola_ROI setovi" \\
      --output data/yolo_imagej
  
  # Custom split ratios
  python scripts/imagej_rois_to_yolo.py \\
      --rois-csv data/annotations/imagej_rois_bez.csv \\
      --images-dir "data/training_data/Collembola_ROI setovi" \\
      --output data/yolo_imagej \\
      --train-ratio 0.8 --val-ratio 0.1
        """
    )
    
    parser.add_argument(
        '--rois-csv',
        type=Path,
        default=Path('data/annotations/imagej_rois_bez.csv'),
        help='Path to ImageJ ROIs CSV file (from convert_imagej_rois.py)'
    )
    
    parser.add_argument(
        '--images-dir',
        type=Path,
        default=Path('data/training_data/Collembola_ROI setovi'),
        help='Base directory containing plate images'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/yolo_imagej'),
        help='Output directory for YOLO dataset'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training split ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.rois_csv.exists():
        print(f"ERROR: ROIs CSV not found: {args.rois_csv}")
        print(f"Please run: python scripts/convert_imagej_rois.py first")
        sys.exit(1)
    
    if not args.images_dir.exists():
        print(f"ERROR: Images directory not found: {args.images_dir}")
        sys.exit(1)
    
    # Convert to YOLO
    prepare_yolo_from_imagej_rois(
        rois_csv=args.rois_csv,
        images_base_dir=args.images_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
