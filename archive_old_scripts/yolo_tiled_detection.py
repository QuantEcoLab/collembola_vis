#!/usr/bin/env python3
"""
YOLO detection with tiling for very large images.
Divides large images into tiles, runs YOLO on each tile at native resolution,
then merges detections with Non-Maximum Suppression (NMS) to remove duplicates.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import pandas as pd


def create_tiles(image_shape: Tuple[int, int], tile_size: int = 1280, overlap: int = 128) -> List[Dict]:
    """
    Create tile coordinates for a large image with overlap.
    
    Args:
        image_shape: (height, width) of the full image
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles to avoid missing objects at boundaries
        
    Returns:
        List of tile dictionaries with coordinates
    """
    height, width = image_shape
    tiles = []
    
    # Calculate step size (tile_size - overlap)
    step = tile_size - overlap
    
    y_positions = list(range(0, height - tile_size + 1, step))
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    
    x_positions = list(range(0, width - tile_size + 1, step))
    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    
    for y in y_positions:
        for x in x_positions:
            tiles.append({
                'x': x,
                'y': y,
                'w': tile_size,
                'h': tile_size,
                'x2': min(x + tile_size, width),
                'y2': min(y + tile_size, height)
            })
    
    return tiles


def non_max_suppression_custom(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply NMS to remove duplicate detections from overlapping tiles.
    
    Args:
        detections: List of detection dictionaries with keys: x1, y1, x2, y2, conf, cls
        iou_threshold: IoU threshold for considering detections as duplicates
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    
    while len(detections) > 0:
        # Take the detection with highest confidence
        best = detections[0]
        keep.append(best)
        detections = detections[1:]
        
        # Remove detections that overlap significantly with the best one
        filtered = []
        for det in detections:
            iou = compute_iou(best, det)
            if iou < iou_threshold:
                filtered.append(det)
        
        detections = filtered
    
    return keep


def compute_iou(det1: Dict, det2: Dict) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    # Calculate intersection
    x1 = max(det1['x1'], det2['x1'])
    y1 = max(det1['y1'], det2['y1'])
    x2 = min(det1['x2'], det2['x2'])
    y2 = min(det1['y2'], det2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (det1['x2'] - det1['x1']) * (det1['y2'] - det1['y1'])
    area2 = (det2['x2'] - det2['x1']) * (det2['y2'] - det2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def detect_on_tiled_image(
    image_path: str,
    model_path: str,
    tile_size: int = 1280,
    overlap: int = 128,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    output_dir: str = None,
    visualize: bool = True,
) -> List[Dict]:
    """
    Run YOLO detection on a large image using tiling approach.
    
    Args:
        image_path: Path to the input image
        model_path: Path to YOLO model weights
        tile_size: Size of each tile
        overlap: Overlap between tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        output_dir: Directory to save outputs
        visualize: Whether to create visualization
        
    Returns:
        List of final detections after NMS
    """
    print(f"\n{'='*60}")
    print(f"Tiled YOLO Detection")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Tile size: {tile_size}px, Overlap: {overlap}px")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = img.shape[:2]
    print(f"Image size: {w}×{h} pixels")
    
    # Create tiles
    tiles = create_tiles((h, w), tile_size, overlap)
    print(f"Created {len(tiles)} tiles")
    
    # Load YOLO model
    print(f"\nLoading YOLO model...")
    model = YOLO(model_path)
    
    # Run detection on each tile
    all_detections = []
    
    for idx, tile in enumerate(tiles):
        # Extract tile from image
        tile_img = img[tile['y']:tile['y2'], tile['x']:tile['x2']]
        
        # Run YOLO on tile
        results = model(tile_img, conf=conf_threshold, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates (relative to tile)
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Convert to global image coordinates
                x1 = int(xyxy[0]) + tile['x']
                y1 = int(xyxy[1]) + tile['y']
                x2 = int(xyxy[2]) + tile['x']
                y2 = int(xyxy[3]) + tile['y']
                
                all_detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'conf': conf,
                    'cls': cls,
                    'tile_idx': idx
                })
        
        if (idx + 1) % 10 == 0 or idx == len(tiles) - 1:
            print(f"  Processed {idx + 1}/{len(tiles)} tiles, found {len(all_detections)} raw detections")
    
    print(f"\nTotal raw detections: {len(all_detections)}")
    
    # Apply NMS to remove duplicates
    print(f"Applying NMS (IoU threshold = {iou_threshold})...")
    final_detections = non_max_suppression_custom(all_detections, iou_threshold)
    print(f"Final detections after NMS: {len(final_detections)}")
    
    # Calculate statistics
    if final_detections:
        confs = [d['conf'] for d in final_detections]
        print(f"\nConfidence statistics:")
        print(f"  Min:    {min(confs):.4f}")
        print(f"  Max:    {max(confs):.4f}")
        print(f"  Mean:   {np.mean(confs):.4f}")
        print(f"  Median: {np.median(confs):.4f}")
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # Save detections as CSV
        csv_path = output_dir / f"{image_name}_detections.csv"
        df_data = []
        for i, det in enumerate(final_detections):
            cx = (det['x1'] + det['x2']) / 2
            cy = (det['y1'] + det['y2']) / 2
            w = det['x2'] - det['x1']
            h = det['y2'] - det['y1']
            df_data.append({
                'id': i,
                'x': cx,
                'y': cy,
                'w': w,
                'h': h,
                'confidence': det['conf'],
                'class': det['cls']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detections to: {csv_path}")
        
        # Create visualization
        if visualize:
            vis_img = img.copy()
            
            for det in final_detections:
                # Draw bounding box
                cv2.rectangle(vis_img, (det['x1'], det['y1']), (det['x2'], det['y2']), 
                            (0, 255, 0), 2)
                
                # Draw small confidence label
                label = f"{det['conf']:.2f}"
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
                
                # Background for text
                cv2.rectangle(vis_img, (det['x1'], det['y1'] - text_h - 4), 
                            (det['x1'] + text_w, det['y1']), (0, 255, 0), -1)
                
                # Text
                cv2.putText(vis_img, label, (det['x1'], det['y1'] - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            vis_path = output_dir / f"{image_name}_detections.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            print(f"Saved visualization to: {vis_path}")
    
    return final_detections


def main():
    parser = argparse.ArgumentParser(description='YOLO detection with tiling for large images')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to YOLO model weights')
    parser.add_argument('--tile-size', type=int, default=1280, help='Tile size in pixels')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap between tiles')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--output', default='outputs/tiled_yolo', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    detections = detect_on_tiled_image(
        image_path=args.image,
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
        visualize=not args.no_viz,
    )
    
    print(f"\n{'='*60}")
    print(f"Detection complete! Found {len(detections)} organisms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
