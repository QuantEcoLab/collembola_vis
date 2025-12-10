#!/usr/bin/env python3
"""
Prepare YOLO dataset from collembola annotations.

YOLO format:
- Images in: images/train/, images/val/
- Labels in: labels/train/, labels/val/
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


def prepare_yolo_dataset(
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    # test_ratio = 0.15 (remaining)
):
    """
    Prepare YOLO dataset from ground truth annotations.
    
    Strategy:
    - Use K1 and C1 plates for training (598 + 692 = 1290 organisms)
    - Use C5 plate for validation (958 organisms)
    
    This ensures validation on a completely separate plate.
    """
    
    print("="*60)
    print("YOLO Dataset Preparation for Collembola Detection")
    print("="*60)
    print()
    
    # Paths
    data_dir = Path("data")
    images_dir = data_dir / "slike"
    annotations_path = data_dir / "annotations" / "collembolas_table.csv"
    
    # Create output structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    df = pd.read_csv(annotations_path)
    print(f"Total annotations: {len(df)}")
    print()
    
    # Map plate names (CSV has different naming than actual files)
    plate_mapping = {
        'K1': 'K1_Fe2O3001 (1).jpg',
        'C1': 'C1_1_Fe2O3002 (1).jpg',
        'C5': 'C5_2_Fe2O3003 (1).jpg'
    }
    
    # Split strategy: K1+C1 = train, C5 = val
    train_plates = ['K1', 'C1']
    val_plates = ['C5']
    
    stats = {'train': 0, 'val': 0}
    
    for plate_id, image_name in plate_mapping.items():
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue
        
        # Get annotations for this plate
        plate_annots = df[df['id_collembole'].str.contains(plate_id)]
        
        if len(plate_annots) == 0:
            print(f"WARNING: No annotations for {plate_id}")
            continue
        
        # Determine split
        if plate_id in train_plates:
            split = 'train'
        elif plate_id in val_plates:
            split = 'val'
        else:
            continue
        
        # Load image to get dimensions
        img = Image.open(image_path)
        img_w, img_h = img.size
        
        # Copy image
        dest_image = output_dir / "images" / split / image_name
        shutil.copy(image_path, dest_image)
        
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
        print(f"{split.upper()}: {image_name} - {len(yolo_labels)} organisms")
    
    print()
    print(f"Dataset Statistics:")
    print(f"  Train: {stats['train']} organisms")
    print(f"  Val: {stats['val']} organisms")
    print()
    
    # Create YOLO data.yaml configuration
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes
        'names': ['collembola']  # Class names
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created YOLO configuration: {yaml_path}")
    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/  ({stats['train']} organisms across {len(train_plates)} plates)")
    print(f"    │   └── val/    ({stats['val']} organisms across {len(val_plates)} plates)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/")
    print(f"    │   └── val/")
    print(f"    └── data.yaml")
    print()
    print("✓ Dataset ready for YOLO training!")
    print()
    print("Next steps:")
    print("  1. Train: yolo detect train data=data/yolo/data.yaml model=yolov8n.pt epochs=100")
    print("  2. Validate: yolo detect val model=runs/detect/train/weights/best.pt")
    print("  3. Predict: yolo detect predict model=runs/detect/train/weights/best.pt source=data/slike/")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--output", type=Path, default=Path("data/yolo"), help="Output directory")
    args = parser.parse_args()
    
    prepare_yolo_dataset(args.output)
