#!/usr/bin/env python3
"""
Visualize YOLO detections with customizable bounding box appearance.
Allows small, clean visualization to assess detection quality.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

def visualize_yolo_detections(
    image_path,
    label_path,
    output_path,
    conf_threshold=0.0,
    box_thickness=2,
    font_scale=0.4,
    show_conf=True,
    show_index=False,
    box_color=(0, 255, 0),
):
    """
    Visualize YOLO detections with minimal clutter.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Read detections
    detections = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    if conf >= conf_threshold:
                        detections.append({
                            'cls': cls_id,
                            'x': x_center,
                            'y': y_center,
                            'w': width,
                            'h': height,
                            'conf': conf
                        })
    
    print(f"Found {len(detections)} detections (conf >= {conf_threshold})")
    
    # Draw detections
    for idx, det in enumerate(detections):
        # Convert normalized coords to pixel coords
        x_center = int(det['x'] * w)
        y_center = int(det['y'] * h)
        box_w = int(det['w'] * w)
        box_h = int(det['h'] * h)
        
        # Calculate top-left corner
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Prepare label text
        label_parts = []
        if show_index:
            label_parts.append(f"#{idx+1}")
        if show_conf:
            label_parts.append(f"{det['conf']:.3f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get text size to create background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Position label above box (or below if too close to top)
            if y1 - text_h - 5 > 0:
                text_y = y1 - 5
                bg_y1 = y1 - text_h - 8
                bg_y2 = y1 - 2
            else:
                text_y = y2 + text_h + 5
                bg_y1 = y2 + 2
                bg_y2 = y2 + text_h + 8
            
            # Draw semi-transparent background for text
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, bg_y1), (x1 + text_w + 4, bg_y2), box_color, -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
            
            # Draw text
            cv2.putText(
                img, label, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )
    
    # Save output
    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization to {output_path}")
    
    # Print statistics
    if detections:
        confs = [d['conf'] for d in detections]
        print(f"\nConfidence statistics:")
        print(f"  Min: {min(confs):.4f}")
        print(f"  Max: {max(confs):.4f}")
        print(f"  Mean: {np.mean(confs):.4f}")
        print(f"  Median: {np.median(confs):.4f}")


if __name__ == "__main__":
    # Example usage for low-confidence detections
    image_path = "data/slike/K1_Fe2O3001 (1).jpg"
    label_path = "outputs/yolo_1280_lowconf/labels/K1_Fe2O3001 (1).txt"
    output_path = "outputs/yolo_1280_lowconf_clean.jpg"
    
    if len(sys.argv) >= 4:
        image_path = sys.argv[1]
        label_path = sys.argv[2]
        output_path = sys.argv[3]
    
    visualize_yolo_detections(
        image_path=image_path,
        label_path=label_path,
        output_path=output_path,
        conf_threshold=0.01,
        box_thickness=1,
        font_scale=0.3,
        show_conf=True,
        show_index=False,
        box_color=(0, 255, 0),
    )
