"""
Complete collembola detection pipeline.

End-to-end detection orchestration:
1. Region proposal (SAM or CV-based)
   - SAM: High accuracy, matches training distribution (~5 min/plate)
   - CV: 70x faster, lower recall (~5 sec/plate)
2. Batch CNN classification (GPU-accelerated)
3. Watershed segmentation refinement (optional)
4. Morphological measurements
5. Export CSV + overlay visualization

Target performance: 80%+ recall, 75%+ precision
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from .proposal_cv import propose_regions_cv, RegionProposal
from .proposal_cv_fast import propose_regions_cv_fast
from .proposal_sam import propose_regions_sam_fast
from .classify_batch import filter_proposals_by_classification
from .segment_watershed import refine_proposals_watershed
from .morphology import mask_measurements
from .config import CLASSIFIER_THRESHOLD


def detect_organisms(
    image_path: Path | str,
    output_dir: Optional[Path | str] = None,
    proposal_method: str = 'sam',  # 'sam' or 'cv'
    use_watershed: bool = False,
    use_grabcut: bool = False,
    confidence_threshold: float = CLASSIFIER_THRESHOLD,
    batch_size: int = 64,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Detect collembola organisms in a plate image.
    
    Args:
        image_path: Path to input plate image
        output_dir: Directory to save outputs (CSV + overlay). If None, no save.
        proposal_method: 'sam' (accurate) or 'cv' (fast)
        use_watershed: Apply watershed refinement to masks
        use_grabcut: Use GrabCut instead of watershed (slower, better)
        confidence_threshold: CNN classification threshold
        batch_size: Batch size for CNN inference
        device: 'cuda' or 'cpu'
        verbose: Print progress messages
        
    Returns:
        detections_df: DataFrame with organism measurements
        overlay_image: Visualization with bboxes/masks drawn
    """
    start_time = time.time()
    
    # Load image
    image_path = Path(image_path)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*60}")
    
    image = np.array(Image.open(image_path).convert('RGB'))
    
    if verbose:
        print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Step 1: Region proposal (SAM or CV)
    if verbose:
        method_name = "SAM" if proposal_method == 'sam' else "CV Fast"
        print(f"\n[1/4] {method_name} Region Proposal...")
    
    proposal_start = time.time()
    
    if proposal_method == 'sam':
        proposals = propose_regions_sam_fast(
            image,
            device=device,
            min_area=1000,  # Match GT distribution (median ~8400 px)
            max_area=100000,
            verbose=verbose
        )
    else:  # cv (default - fast dilated CV)
        proposals, plate_circle = propose_regions_cv_fast(
            image,
            bbox_scale_factor=4.5,  # Expand to match GT median area
            min_area=1000,  # Match GT bbox sizes
            max_area=100000,
            min_eccentricity=0.60,  # Relaxed from 0.70
            iou_threshold=0.3,  # Merge overlapping proposals
            detect_plate=True,  # Enable plate detection and masking
            plate_shrink=50,  # Shrink mask to avoid plate edges
            verbose=verbose
        )
        if verbose and plate_circle:
            print(f"  Detected plate: center=({plate_circle[0]}, {plate_circle[1]}), radius={plate_circle[2]}")
    
    proposal_time = time.time() - proposal_start
    
    if verbose:
        print(f"  Found {len(proposals)} proposals in {proposal_time:.2f}s")
    
    if len(proposals) == 0:
        if verbose:
            print("No proposals found. Exiting.")
        empty_df = pd.DataFrame()
        return empty_df, image
    
    # Step 2: Batch CNN classification
    if verbose:
        print(f"\n[2/4] CNN Classification (batch_size={batch_size}, device={device})...")
    
    classify_start = time.time()
    accepted_proposals, confidences = filter_proposals_by_classification(
        image,
        proposals,
        threshold=confidence_threshold,
        batch_size=batch_size,
        device=device,
        verbose=verbose
    )
    classify_time = time.time() - classify_start
    
    if verbose:
        print(f"  Accepted {len(accepted_proposals)}/{len(proposals)} proposals in {classify_time:.2f}s")
    
    if len(accepted_proposals) == 0:
        if verbose:
            print("No organisms detected. Exiting.")
        empty_df = pd.DataFrame()
        return empty_df, image
    
    # Step 3: Watershed refinement (optional)
    watershed_time = 0.0
    if use_watershed or use_grabcut:
        if verbose:
            method = "GrabCut" if use_grabcut else "Watershed"
            print(f"\n[3/4] {method} Segmentation Refinement...")
        
        watershed_start = time.time()
        accepted_proposals = refine_proposals_watershed(
            image,
            accepted_proposals,
            use_grabcut=use_grabcut,
            verbose=verbose
        )
        watershed_time = time.time() - watershed_start
        
        if verbose:
            print(f"  Refined {len(accepted_proposals)} masks in {watershed_time:.2f}s")
    else:
        if verbose:
            print(f"\n[3/4] Skipping watershed (using bbox masks)")
    
    # Step 4: Extract morphological measurements
    if verbose:
        print(f"\n[4/4] Extracting Morphology...")
    
    morph_start = time.time()
    detections = []
    
    for i, (prop, conf) in enumerate(zip(accepted_proposals, confidences)):
        # Get measurements from mask
        measurements = mask_measurements(prop.mask)
        
        if not measurements:
            continue
        
        x, y, w, h = prop.bbox
        
        detection = {
            'organism_id': i,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'confidence': conf,
            **measurements
        }
        
        detections.append(detection)
    
    morph_time = time.time() - morph_start
    
    if verbose:
        print(f"  Measured {len(detections)} organisms in {morph_time:.2f}s")
    
    # Create DataFrame
    detections_df = pd.DataFrame(detections)
    
    # Generate overlay visualization
    overlay = create_overlay(image, accepted_proposals, confidences)
    
    # Summary
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Detected organisms: {len(detections)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"  - Proposal: {proposal_time:.2f}s")
        print(f"  - Classification: {classify_time:.2f}s")
        if use_watershed or use_grabcut:
            print(f"  - Segmentation: {watershed_time:.2f}s")
        print(f"  - Morphology: {morph_time:.2f}s")
        print(f"{'='*60}\n")
    
    # Save outputs if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        stem = image_path.stem
        csv_path = output_dir / f"{stem}_organisms.csv"
        detections_df.to_csv(csv_path, index=False)
        
        # Save overlay
        overlay_path = output_dir / f"{stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        
        if verbose:
            print(f"Saved outputs:")
            print(f"  CSV: {csv_path}")
            print(f"  Overlay: {overlay_path}")
    
    return detections_df, overlay


def create_overlay(
    image: np.ndarray,
    proposals: List[RegionProposal],
    confidences: List[float],
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create visualization overlay with bboxes and masks.
    
    Args:
        image: RGB image (H, W, 3)
        proposals: List of RegionProposal objects
        confidences: Classification confidences
        alpha: Transparency for masks
        
    Returns:
        Overlay image with annotations
    """
    overlay = image.copy()
    
    # Color map: green for high confidence, yellow for medium, orange for low
    for prop, conf in zip(proposals, confidences):
        # Color based on confidence
        if conf >= 0.99:
            color = (0, 255, 0)  # Green
        elif conf >= 0.98:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw bounding box
        x, y, w, h = prop.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # Draw mask overlay (semi-transparent)
        mask_color = np.zeros_like(image)
        mask_color[prop.mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0)
        
        # Draw confidence text
        text = f"{conf:.3f}"
        cv2.putText(
            overlay,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    
    return overlay


def batch_process_plates(
    image_paths: List[Path | str],
    output_dir: Path | str,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Process multiple plates in batch.
    
    Args:
        image_paths: List of image paths
        output_dir: Directory for outputs
        **kwargs: Arguments passed to detect_organisms
        
    Returns:
        Dictionary mapping image name to detections DataFrame
    """
    output_dir = Path(output_dir)
    results = {}
    
    print(f"\nProcessing {len(image_paths)} plates...")
    print(f"Output directory: {output_dir}\n")
    
    for i, image_path in enumerate(image_paths, 1):
        image_path = Path(image_path)
        print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
        
        try:
            detections_df, _ = detect_organisms(
                image_path,
                output_dir=output_dir,
                **kwargs
            )
            results[image_path.stem] = detections_df
        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")
            results[image_path.stem] = pd.DataFrame()
    
    # Create summary
    summary = {
        'plate': [],
        'organisms': [],
        'avg_length_um': [],
        'avg_width_um': [],
        'avg_volume_um3': []
    }
    
    for plate_name, df in results.items():
        summary['plate'].append(plate_name)
        summary['organisms'].append(len(df))
        
        if len(df) > 0:
            summary['avg_length_um'].append(df['length_um'].mean())
            summary['avg_width_um'].append(df['width_um'].mean())
            summary['avg_volume_um3'].append(df['volume_um3'].mean())
        else:
            summary['avg_length_um'].append(0)
            summary['avg_width_um'].append(0)
            summary['avg_volume_um3'].append(0)
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    
    return results
