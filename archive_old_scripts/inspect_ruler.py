#!/usr/bin/env python3
"""
Visual inspection script to help understand ruler location and appearance.

This script:
1. Extracts all 4 corners of an image
2. Saves them as separate files for manual inspection
3. Shows edge detection results to help tune parameters

Usage:
    python scripts/inspect_ruler.py --image data/slike/K1_Fe2O3001\ \(1\).jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def inspect_image_corners(image_path: Path, 
                          corner_size: int = 1500,
                          output_dir: Path = Path('ruler_inspection')):
    """
    Extract and save all corners for manual inspection.
    """
    print(f"Loading image: {image_path}")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    print(f"Image size: {img.width} × {img.height}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define corners
    corners = {
        'top_left': (0, 0, corner_size, corner_size),
        'top_right': (img.width - corner_size, 0, img.width, corner_size),
        'bottom_left': (0, img.height - corner_size, corner_size, img.height),
        'bottom_right': (img.width - corner_size, img.height - corner_size, 
                        img.width, img.height),
    }
    
    print(f"\nExtracting corners (size: {corner_size}×{corner_size})...")
    
    for name, bbox in corners.items():
        # Extract corner
        crop = img.crop(bbox)
        
        # Save original
        crop_path = output_dir / f"{name}_original.jpg"
        crop.save(crop_path, quality=95)
        print(f"✓ Saved: {crop_path}")
        
        # Convert to numpy for edge detection
        crop_np = np.array(crop.convert('RGB'))
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Save edges
        edges_path = output_dir / f"{name}_edges.jpg"
        cv2.imwrite(str(edges_path), edges)
        print(f"✓ Saved edges: {edges_path}")
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=100, 
                                minLineLength=200, 
                                maxLineGap=20)
        
        # Draw lines on image
        line_img = crop_np.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            lines_path = output_dir / f"{name}_lines.jpg"
            cv2.imwrite(str(lines_path), cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved lines: {lines_path} ({len(lines)} lines detected)")
        else:
            print(f"  ⚠ No lines detected in {name}")
    
    print(f"\n{'='*70}")
    print(f"INSPECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Files per corner:")
    print(f"  - *_original.jpg  : Raw corner crop")
    print(f"  - *_edges.jpg     : Canny edge detection")
    print(f"  - *_lines.jpg     : Hough line detection")
    print(f"\nNEXT STEPS:")
    print(f"1. Open the output folder and examine all images")
    print(f"2. Identify which corner contains the ruler")
    print(f"3. Look at edge/line detection to see if ruler is being detected")
    print(f"4. If ruler not visible, try larger corner_size")
    print(f"5. If edges not detecting ruler, adjust Canny thresholds")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect image corners to locate ruler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to microscope image')
    parser.add_argument('--corner-size', type=int, default=1500,
                        help='Corner crop size in pixels (default: 1500)')
    parser.add_argument('--output', type=str, default='ruler_inspection',
                        help='Output directory (default: ruler_inspection)')
    
    args = parser.parse_args()
    
    # Validate image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Run inspection
    inspect_image_corners(
        image_path=image_path,
        corner_size=args.corner_size,
        output_dir=Path(args.output)
    )


if __name__ == '__main__':
    main()
