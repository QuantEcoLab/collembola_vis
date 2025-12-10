#!/usr/bin/env python3
"""
Convert ImageJ ROI zip files to CSV format.

This script reads ImageJ ROI Set .zip files (containing multiple .roi files)
and extracts bounding box annotations to a unified CSV format compatible with
the rest of the collembola pipeline.

Usage:
    python scripts/convert_imagej_rois.py --roi-dir "data/training_data/Collembola_ROI setovi" --output data/annotations/imagej_rois.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from read_roi import read_roi_zip
from typing import List, Dict
import argparse


def extract_rois_from_zip(zip_path: Path, image_name: str, plate_id: str) -> List[Dict]:
    """
    Extract ROI annotations from ImageJ zip file.
    
    Args:
        zip_path: Path to RoiSet .zip file
        image_name: Associated image filename
        plate_id: Plate identifier (e.g., 'K_1_Collembola001')
    
    Returns:
        List of annotation dictionaries with keys: plate_id, image_name, x, y, w, h, roi_name
    """
    try:
        rois = read_roi_zip(str(zip_path))
    except Exception as e:
        print(f"ERROR reading {zip_path}: {e}")
        return []
    
    annotations = []
    for roi_name, roi_data in rois.items():
        # Extract bounding box coordinates
        # ImageJ ROI format uses 'left', 'top', 'width', 'height'
        if 'left' in roi_data and 'top' in roi_data and 'width' in roi_data and 'height' in roi_data:
            annotations.append({
                'plate_id': plate_id,
                'image_name': image_name,
                'x': roi_data['left'],
                'y': roi_data['top'],
                'w': roi_data['width'],
                'h': roi_data['height'],
                'roi_name': roi_name,
                'roi_type': roi_data.get('type', 'unknown')
            })
        else:
            print(f"WARNING: ROI {roi_name} in {zip_path} missing bbox coordinates")
    
    return annotations


def find_bez_roi_files(roi_base_dir: Path) -> List[Dict]:
    """
    Find all 'bez antene i furce' ROI zip files.
    
    Args:
        roi_base_dir: Base directory containing ROI setovi
    
    Returns:
        List of dicts with keys: roi_zip_path, image_path, plate_id
    """
    roi_files = []
    
    # Maxima ROI setovi (Fe2O3 treatment)
    maxima_dir = roi_base_dir / "Maxima_ROI setovi" / "Fe2O3"
    if maxima_dir.exists():
        for plate_dir in sorted(maxima_dir.glob("ROI_*")):
            plate_id = plate_dir.name.replace("ROI_", "")
            
            # Find the 'bez antene i furce' zip file
            bez_zips = list(plate_dir.glob("*bez antene i furce.zip"))
            if not bez_zips:
                print(f"WARNING: No 'bez antene' zip found in {plate_dir}")
                continue
            
            roi_zip = bez_zips[0]
            
            # Find corresponding image
            image_files = list(plate_dir.glob("*.jpg")) + list(plate_dir.glob("*.JPG"))
            if not image_files:
                print(f"WARNING: No image found in {plate_dir}")
                continue
            
            image_path = image_files[0]
            
            roi_files.append({
                'roi_zip_path': roi_zip,
                'image_path': image_path,
                'plate_id': plate_id
            })
    
    # Luca ROI setovi (Mikroplastika treatment)
    luca_dir = roi_base_dir / "Luca_ROI setovi" / "Mikroplastika"
    if luca_dir.exists():
        for plate_dir in sorted(luca_dir.glob("C_*")):
            plate_id = plate_dir.name
            
            # Find any zip file (Luca's don't have the 'bez/sa' distinction)
            zip_files = list(plate_dir.glob("*.zip"))
            if not zip_files:
                print(f"WARNING: No zip found in {plate_dir}")
                continue
            
            roi_zip = zip_files[0]
            
            # Find corresponding image
            image_files = list(plate_dir.glob("*.jpg")) + list(plate_dir.glob("*.JPG"))
            if not image_files:
                print(f"WARNING: No image found in {plate_dir}")
                continue
            
            image_path = image_files[0]
            
            roi_files.append({
                'roi_zip_path': roi_zip,
                'image_path': image_path,
                'plate_id': plate_id
            })
    
    return roi_files


def convert_imagej_rois_to_csv(roi_base_dir: Path, output_csv: Path, verbose: bool = True):
    """
    Convert all ImageJ ROI zip files to unified CSV format.
    
    Args:
        roi_base_dir: Base directory containing "Maxima_ROI setovi" and "Luca_ROI setovi"
        output_csv: Output CSV file path
        verbose: Print progress messages
    """
    print("="*70)
    print("ImageJ ROI to CSV Converter")
    print("="*70)
    print()
    
    # Find all 'bez antene i furce' ROI files
    roi_files = find_bez_roi_files(roi_base_dir)
    
    if not roi_files:
        print("ERROR: No ROI files found!")
        return
    
    print(f"Found {len(roi_files)} ROI zip files to process")
    print()
    
    # Extract all annotations
    all_annotations = []
    total_rois = 0
    
    for roi_info in roi_files:
        roi_zip = roi_info['roi_zip_path']
        image_path = roi_info['image_path']
        plate_id = roi_info['plate_id']
        
        if verbose:
            print(f"Processing: {plate_id}")
            print(f"  ROI file: {roi_zip.name}")
            print(f"  Image: {image_path.name}")
        
        annotations = extract_rois_from_zip(roi_zip, image_path.name, plate_id)
        
        if verbose:
            print(f"  ✓ Extracted {len(annotations)} ROIs")
            print()
        
        all_annotations.extend(annotations)
        total_rois += len(annotations)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_annotations)
    
    # Reorder columns for clarity
    df = df[['plate_id', 'image_name', 'x', 'y', 'w', 'h', 'roi_name', 'roi_type']]
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("="*70)
    print("✓ Conversion Complete!")
    print("="*70)
    print(f"Total plates: {len(roi_files)}")
    print(f"Total ROIs extracted: {total_rois}")
    print(f"Output saved to: {output_csv}")
    print()
    
    # Summary statistics
    print("ROIs per plate:")
    plate_counts = df.groupby('plate_id').size().sort_values(ascending=False)
    for plate, count in plate_counts.items():
        print(f"  {plate}: {count}")
    print()
    
    print("ROI type distribution:")
    type_counts = df['roi_type'].value_counts()
    for roi_type, count in type_counts.items():
        print(f"  {roi_type}: {count}")
    print()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert ImageJ ROI zip files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all 'bez antene i furce' ROIs to CSV
  python scripts/convert_imagej_rois.py \\
      --roi-dir "data/training_data/Collembola_ROI setovi" \\
      --output data/annotations/imagej_rois_bez.csv
  
  # Quiet mode (less output)
  python scripts/convert_imagej_rois.py \\
      --roi-dir "data/training_data/Collembola_ROI setovi" \\
      --output data/annotations/imagej_rois_bez.csv \\
      --quiet
        """
    )
    
    parser.add_argument(
        '--roi-dir',
        type=Path,
        default=Path('data/training_data/Collembola_ROI setovi'),
        help='Base directory containing ROI setovi (default: data/training_data/Collembola_ROI setovi)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/annotations/imagej_rois_bez.csv'),
        help='Output CSV file path (default: data/annotations/imagej_rois_bez.csv)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Convert ROIs to CSV
    df = convert_imagej_rois_to_csv(
        roi_base_dir=args.roi_dir,
        output_csv=args.output,
        verbose=not args.quiet
    )
    
    print(f"✓ CSV dataset ready: {args.output}")


if __name__ == '__main__':
    main()
