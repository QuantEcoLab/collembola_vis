#!/usr/bin/env python3
"""
Create tiled YOLO dataset from large images with ImageJ ROI annotations.

This script:
1. Tiles large images (10K x 10K) into smaller tiles (e.g., 1280x1280)
2. Maps ROI annotations to tiles with overlap handling
3. Creates YOLO format dataset with train/val/test splits
4. Preserves tile metadata for inference reconstruction
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import random

# Configuration
TILE_SIZE = 1280  # Size of each tile
OVERLAP = 256     # Overlap between tiles to avoid edge issues
MIN_BOX_AREA_RATIO = 0.3  # Min % of box that must be in tile to include

# Paths
ROI_CSV = "data/annotations/imagej_rois_bez.csv"
IMAGE_BASE = "data/training_data/Collembola_ROI setovi/Maxima_ROI setovi/Fe2O3"
OUTPUT_BASE = "data/yolo_tiled"
METADATA_FILE = os.path.join(OUTPUT_BASE, "tile_metadata.json")

# Dataset splits (same plates as before)
TRAIN_PLATES = [
    'K_1_Collembola001', 'K_2_Collembola002', 'K_3_Collembola003',
    'C_1_1_Collembola004', 'C_1_2_Collembola005', 'C_1_3_Collembola006',
    'C_2_1_Collembola007', 'C_2_2_Collembola008', 'C_2_3_Collembola009',
    'C_3_1_Collembola010', 'C_3_2_Collembola011', 'C_3_3_Collembola012',
    'C_4_2_Collembola014'
]

VAL_PLATES = ['C_4_1_Collembola013', 'C_4_3_Collembola015']

TEST_PLATES = []  # Will be determined from available plates


def tile_image(image_path, tile_size=1280, overlap=256):
    """
    Tile a large image into smaller overlapping tiles.
    
    Returns:
        List of (tile_img, x_offset, y_offset, tile_id)
    """
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    tiles = []
    stride = tile_size - overlap
    
    tile_id = 0
    y = 0
    y_end = 0
    while y < img_h:
        x = 0
        x_end = 0
        while x < img_w:
            # Adjust tile boundaries at edges
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size) if x_end == img_w else x
            y_start = max(0, y_end - tile_size) if y_end == img_h else y
            
            # Crop tile
            tile = img.crop((x_start, y_start, x_end, y_end))
            tiles.append((tile, x_start, y_start, tile_id))
            tile_id += 1
            
            if x_end == img_w:
                break
            x += stride
        
        if y_end == img_h:
            break
        y += stride
    
    return tiles


def map_roi_to_tile(roi_x, roi_y, roi_w, roi_h, tile_x, tile_y, tile_size):
    """
    Map a ROI from full image coordinates to tile coordinates.
    
    Returns:
        (x_center, y_center, width, height) in normalized YOLO format,
        or None if ROI doesn't sufficiently overlap with tile.
    """
    # ROI bounds in full image
    roi_x1, roi_y1 = roi_x, roi_y
    roi_x2, roi_y2 = roi_x + roi_w, roi_y + roi_h
    
    # Tile bounds in full image
    tile_x1, tile_y1 = tile_x, tile_y
    tile_x2, tile_y2 = tile_x + tile_size, tile_y + tile_size
    
    # Find intersection
    inter_x1 = max(roi_x1, tile_x1)
    inter_y1 = max(roi_y1, tile_y1)
    inter_x2 = min(roi_x2, tile_x2)
    inter_y2 = min(roi_y2, tile_y2)
    
    # Check if there's overlap
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return None
    
    # Check if enough of the box is in the tile
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    roi_area = roi_w * roi_h
    
    if inter_area / roi_area < MIN_BOX_AREA_RATIO:
        return None
    
    # Convert to tile-relative coordinates
    box_x1 = max(0, roi_x1 - tile_x1)
    box_y1 = max(0, roi_y1 - tile_y1)
    box_x2 = min(tile_size, roi_x2 - tile_x1)
    box_y2 = min(tile_size, roi_y2 - tile_y1)
    
    # Convert to YOLO format (normalized center, width, height)
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1
    center_x = (box_x1 + box_x2) / 2.0 / tile_size
    center_y = (box_y1 + box_y2) / 2.0 / tile_size
    norm_w = box_w / tile_size
    norm_h = box_h / tile_size
    
    return (center_x, center_y, norm_w, norm_h)


def create_tiled_dataset():
    """Main function to create tiled YOLO dataset."""
    
    # Load ROI annotations
    print("Loading ROI annotations...")
    rois_df = pd.read_csv(ROI_CSV)
    print(f"Loaded {len(rois_df)} ROI annotations from {rois_df['plate_id'].nunique()} plates")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, 'labels', split), exist_ok=True)
    
    # Track metadata
    tile_metadata = {}
    stats = defaultdict(lambda: {'tiles': 0, 'annotations': 0})
    
    # Group ROIs by plate
    grouped_rois = rois_df.groupby('plate_id')
    
    # Determine all available plates
    all_plates = list(grouped_rois.groups.keys())
    test_plates = [p for p in all_plates if p not in TRAIN_PLATES and p not in VAL_PLATES]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(TRAIN_PLATES)} plates")
    print(f"  Val:   {len(VAL_PLATES)} plates")
    print(f"  Test:  {len(test_plates)} plates")
    
    # Process each plate
    for plate_id, plate_rois in grouped_rois:
        # Determine split
        if plate_id in TRAIN_PLATES:
            split = 'train'
        elif plate_id in VAL_PLATES:
            split = 'val'
        else:
            split = 'test'
        
        # Find image path
        image_name = plate_rois.iloc[0]['image_name']
        image_path = os.path.join(IMAGE_BASE, f"ROI_{plate_id}", image_name)
        
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            continue
        
        print(f"\nProcessing {plate_id} ({split}) - {len(plate_rois)} ROIs...")
        
        # Tile the image
        tiles = tile_image(image_path, TILE_SIZE, OVERLAP)
        print(f"  Created {len(tiles)} tiles")
        
        # Process each tile
        for tile_img, tile_x, tile_y, tile_id in tiles:
            tile_name = f"{plate_id}_tile_{tile_id:04d}"
            
            # Map ROIs to this tile
            yolo_annotations = []
            for _, roi in plate_rois.iterrows():
                mapped = map_roi_to_tile(
                    roi['x'], roi['y'], roi['w'], roi['h'],
                    tile_x, tile_y, TILE_SIZE
                )
                if mapped:
                    # Class 0 for collembola
                    cx, cy, w, h = mapped
                    yolo_annotations.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            # Save tile and annotations if it has annotations
            # (or save all tiles to allow background learning)
            if len(yolo_annotations) > 0:  # Only save tiles with annotations
                # Save image
                img_path = os.path.join(OUTPUT_BASE, 'images', split, f"{tile_name}.jpg")
                tile_img.save(img_path, quality=95)
                
                # Save YOLO labels
                label_path = os.path.join(OUTPUT_BASE, 'labels', split, f"{tile_name}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Store metadata
                tile_metadata[tile_name] = {
                    'plate_id': plate_id,
                    'tile_id': tile_id,
                    'offset_x': tile_x,
                    'offset_y': tile_y,
                    'tile_size': TILE_SIZE,
                    'split': split,
                    'num_annotations': len(yolo_annotations)
                }
                
                stats[split]['tiles'] += 1
                stats[split]['annotations'] += len(yolo_annotations)
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(tile_metadata, f, indent=2)
    
    print(f"\nTile metadata saved to {METADATA_FILE}")
    
    # Create YOLO data.yaml
    yaml_content = f"""# Tiled Collembola Dataset
path: {os.path.abspath(OUTPUT_BASE)}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1
names: ['collembola']

# Tiling info
tile_size: {TILE_SIZE}
overlap: {OVERLAP}
"""
    
    yaml_path = os.path.join(OUTPUT_BASE, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO config saved to {yaml_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    for split in ['train', 'val', 'test']:
        print(f"{split.upper():5s}: {stats[split]['tiles']:4d} tiles, {stats[split]['annotations']:5d} annotations")
    print("="*60)


if __name__ == '__main__':
    create_tiled_dataset()
