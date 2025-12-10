#!/usr/bin/env python3
"""
Tiled inference for large collembola images.

This script:
1. Tiles large images with overlap
2. Runs YOLO detection on each tile
3. Merges predictions with NMS across tile boundaries
4. Outputs full-image detections in CSV and overlay visualization

Usage:
    python scripts/infer_tiled.py --image path/to/image.jpg --model runs/detect/train_tiled_1280_20251210_115016/weights/best.pt
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch


def tile_image(image_path, tile_size=1280, overlap=256):
    """
    Tile a large image into smaller overlapping tiles.
    
    Returns:
        img: PIL Image
        tiles: List of (tile_img, x_offset, y_offset, tile_id)
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
    
    return img, tiles


def nms_global(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression across all tiles.
    
    Args:
        detections: List of dicts with keys: x1, y1, x2, y2, conf, class
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of kept detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections])
    scores = np.array([d['conf'] for d in detections])
    
    # Sort by confidence
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Pick detection with highest confidence
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_order - inter
        
        iou = inter / union
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [detections[i] for i in keep]


def infer_tiled(
    image_path,
    model_path,
    tile_size=1280,
    overlap=256,
    conf_threshold=0.25,
    iou_threshold=0.5,
    output_dir=None,
    device='0'
):
    """
    Run tiled inference on a large image.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        tile_size: Size of tiles
        overlap: Overlap between tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        output_dir: Output directory (default: infer_tiled_output)
        device: GPU device(s) to use
    
    Returns:
        detections: List of detection dictionaries
    """
    # Setup
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = Path('infer_tiled_output')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_name = image_path.stem
    
    print(f"Processing: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Tile image
    print(f"\nTiling image (tile_size={tile_size}, overlap={overlap})...")
    img, tiles = tile_image(image_path, tile_size, overlap)
    img_w, img_h = img.size
    print(f"Image size: {img_w}x{img_h}")
    print(f"Created {len(tiles)} tiles")
    
    # Run detection on each tile
    print(f"\nRunning detection on tiles...")
    all_detections = []
    
    for tile_img, x_offset, y_offset, tile_id in tiles:
        # Run YOLO on tile
        results = model(tile_img, conf=conf_threshold, device=device, verbose=False)
        
        # Convert to full image coordinates
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Convert to full image coordinates
                detection = {
                    'x1': float(x1 + x_offset),
                    'y1': float(y1 + y_offset),
                    'x2': float(x2 + x_offset),
                    'y2': float(y2 + y_offset),
                    'conf': conf,
                    'class': cls,
                    'tile_id': tile_id
                }
                all_detections.append(detection)
    
    print(f"Total detections before NMS: {len(all_detections)}")
    
    # Apply global NMS
    print(f"\nApplying global NMS (IoU threshold={iou_threshold})...")
    final_detections = nms_global(all_detections, iou_threshold)
    print(f"Final detections after NMS: {len(final_detections)}")
    
    # Save to CSV
    csv_path = output_dir / f"{image_name}_detections.csv"
    df_data = []
    for det in final_detections:
        df_data.append({
            'x1': det['x1'],
            'y1': det['y1'],
            'x2': det['x2'],
            'y2': det['y2'],
            'width': det['x2'] - det['x1'],
            'height': det['y2'] - det['y1'],
            'confidence': det['conf'],
            'class': det['class']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detections to: {csv_path}")
    
    # Create visualization
    print(f"\nCreating visualization...")
    draw = ImageDraw.Draw(img)
    
    for det in final_detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw confidence score
        text = f"{conf:.2f}"
        draw.text((x1, y1 - 10), text, fill='red')
    
    # Save overlay
    overlay_path = output_dir / f"{image_name}_overlay.jpg"
    img.save(overlay_path, quality=95)
    print(f"Saved overlay to: {overlay_path}")
    
    # Save metadata
    metadata = {
        'image_path': str(image_path),
        'image_size': [img_w, img_h],
        'model_path': str(model_path),
        'tile_size': tile_size,
        'overlap': overlap,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'num_tiles': len(tiles),
        'detections_before_nms': len(all_detections),
        'detections_after_nms': len(final_detections)
    }
    
    metadata_path = output_dir / f"{image_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    print("\n" + "="*70)
    print(f"Inference complete!")
    print(f"Detected {len(final_detections)} collembola organisms")
    print("="*70)
    
    return final_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiled inference for large collembola images')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, 
                        default='runs/detect/train_tiled_1280_20251210_115016/weights/best.pt',
                        help='Path to YOLO model weights')
    parser.add_argument('--tile-size', type=int, default=1280,
                        help='Tile size (default: 1280)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between tiles (default: 256)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--output', type=str, default='infer_tiled_output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device to use (default: 0)')
    
    args = parser.parse_args()
    
    infer_tiled(
        image_path=args.image,
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
        device=args.device
    )
