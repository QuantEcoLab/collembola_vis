#!/usr/bin/env python3
"""
Batch process multiple collembola plate images.

Each image is processed into its own subfolder within the output directory:
  output_dir/
    image1/
      image1_calibration.json
      image1_measurements.csv
      ...
    image2/
      image2_calibration.json
      image2_measurements.csv
      ...

Usage:
    python scripts/process_batch.py --input /home/adeb/data/Collembola* --output /home/adeb/data/Collembola_results
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


def find_images(input_pattern):
    """Find all image files matching pattern."""
    images = []
    
    # Expand glob pattern
    base_path = Path(input_pattern.split('*')[0])
    if '*' in input_pattern:
        pattern = input_pattern.split('/')[-1]
        for p in base_path.parent.glob(pattern):
            if p.is_dir():
                # Search for images in directory
                for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                    images.extend(p.rglob(ext))
    else:
        # Direct path
        p = Path(input_pattern)
        if p.is_dir():
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                images.extend(p.rglob(ext))
        elif p.is_file():
            images.append(p)
    
    return sorted(set(images))


def process_image(image_path, base_output_dir, conf=0.6, default_cal=8.666):
    """Process a single image in its own folder."""
    # Create image-specific output directory
    image_output_dir = base_output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing: {image_path.name}")
    print(f"Output: {image_output_dir}")
    print(f"{'='*80}")
    
    cmd = [
        'python', 'scripts/process_single_image.py',
        str(image_path),
        '--output', str(image_output_dir),
        '--conf', str(conf),
        '--default-cal', str(default_cal)
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = result.returncode == 0
    
    if not success:
        print(f"❌ FAILED: {image_path.name} ({elapsed:.1f}s)")
        print(f"Error: {result.stderr[:500]}")
    else:
        print(f"✅ SUCCESS: {image_path.name} ({elapsed:.1f}s)")
    
    return success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Batch process collembola plate images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory or glob pattern (e.g., /path/Collembola*)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.6,
                       help='YOLO confidence threshold (default: 0.6)')
    parser.add_argument('--default-cal', type=float, default=8.666,
                       help='Default µm/px if calibration fails (default: 8.666)')
    parser.add_argument('--resume', action='store_true',
                       help='Skip images that already have measurements CSV')
    
    args = parser.parse_args()
    
    # Find images
    print(f"Searching for images in: {args.input}")
    images = find_images(args.input)
    
    if not images:
        print(f"ERROR: No images found matching: {args.input}")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Filter images if resuming
    if args.resume:
        remaining = []
        for img in images:
            # Check for measurements CSV in image-specific folder
            image_dir = output_dir / img.stem
            meas_file = image_dir / f"{img.stem}_measurements.csv"
            if not meas_file.exists():
                remaining.append(img)
        
        skipped = len(images) - len(remaining)
        if skipped > 0:
            print(f"Resuming: skipping {skipped} already processed images")
        images = remaining
    
    if not images:
        print("All images already processed!")
        return
    
    print(f"\nProcessing {len(images)} images...")
    print(f"Confidence threshold: {args.conf}")
    print(f"Default calibration: {args.default_cal} µm/px")
    
    # Process images
    start_time = datetime.now()
    results = []
    
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        success, elapsed = process_image(image_path, output_dir, args.conf, args.default_cal)
        results.append({
            'image': image_path.name,
            'success': success,
            'time': elapsed
        })
    
    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total images:     {len(results)}")
    print(f"Successful:       {successful} ✅")
    print(f"Failed:           {failed} ❌")
    print(f"Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average per image: {total_time/len(results):.1f}s")
    print(f"\nResults saved to: {output_dir}")
    
    # Show failed images
    if failed > 0:
        print(f"\nFailed images:")
        for r in results:
            if not r['success']:
                print(f"  - {r['image']}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
