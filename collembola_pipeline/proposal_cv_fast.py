"""
Enhanced CV-based region proposal for collembola detection.

Fast approach:
1. Detect circular plate and mask outside regions
2. Find foreground contours (organism pixels) within plate
3. Expand bboxes by a scale factor (not mask dilation - much faster!)
4. Merge overlapping bboxes via NMS
5. Produces organism-scale bboxes matching ground truth

Speed: ~10-15 seconds (vs 5-7 minutes for SAM)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2

from .proposal_cv import RegionProposal, preprocess_image, subtract_background, threshold_image
from .detect_plate import detect_and_mask_plate


def expand_and_merge_bboxes(
    contours,
    image_shape: Tuple[int, int],
    bbox_scale_factor: float = 3.0,
    min_area: int = 1000,
    max_area: int = 100000,
    min_eccentricity: float = 0.60,
    iou_threshold: float = 0.3,
    verbose: bool = False
) -> List[RegionProposal]:
    """
    Fast bbox expansion without mask dilation.
    
    Strategy:
    - Get bbox from each contour
    - Expand bbox by scale_factor (multiply width/height)
    - Merge overlapping bboxes
    - Much faster than dilating actual masks
    
    Args:
        contours: List of OpenCV contours
        image_shape: (height, width) of image
        bbox_scale_factor: How much to expand bboxes (e.g. 3.0 = 3x larger)
        min_area: Minimum bbox area
        max_area: Maximum bbox area
        min_eccentricity: Minimum eccentricity filter
        iou_threshold: IOU threshold for merging overlapping proposals
        verbose: Print progress
        
    Returns:
        List of RegionProposal objects
    """
    H, W = image_shape
    proposals = []
    
    if verbose:
        print(f"[CV Fast] Processing {len(contours)} contours...")
    
    for contour in contours:
        # Get original bbox
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand bbox by scale_factor
        center_x = x + w / 2
        center_y = y + h / 2
        new_w = w * bbox_scale_factor
        new_h = h * bbox_scale_factor
        
        # Calculate expanded bbox
        new_x = int(center_x - new_w / 2)
        new_y = int(center_y - new_h / 2)
        new_w = int(new_w)
        new_h = int(new_h)
        
        # Clip to image bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(W - new_x, new_w)
        new_h = min(H - new_y, new_h)
        
        bbox_area = new_w * new_h
        
        # Filter by area
        if bbox_area < min_area or bbox_area > max_area:
            continue
        
        # Calculate eccentricity from bbox aspect ratio (approximation)
        if new_h > 0:
            ar = new_w / new_h
            if ar < 1:
                ar = 1 / ar
            # Map aspect ratio to eccentricity
            eccentricity = np.sqrt(1 - (1/ar)**2)
        else:
            eccentricity = 0.5
        
        # Filter by eccentricity
        if eccentricity < min_eccentricity:
            continue
        
        # Create simple bbox mask (for compatibility)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
        
        # Calculate contour area for confidence
        contour_area = cv2.contourArea(contour)
        confidence = min(1.0, contour_area / bbox_area)  # Fill ratio
        
        proposals.append(RegionProposal(
            bbox=(new_x, new_y, new_w, new_h),
            mask=mask,
            confidence=confidence,
            area=bbox_area,
            eccentricity=eccentricity
        ))
    
    if verbose:
        print(f"[CV Fast] After expansion and filtering: {len(proposals)} proposals")
    
    # Non-maximum suppression to merge overlapping
    proposals = non_max_suppression(proposals, iou_threshold, verbose)
    
    if verbose:
        print(f"[CV Fast] After NMS: {len(proposals)} proposals")
    
    # Sort by confidence
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    
    return proposals


def non_max_suppression(
    proposals: List[RegionProposal],
    iou_threshold: float = 0.3,
    verbose: bool = False
) -> List[RegionProposal]:
    """
    Merge overlapping proposals using non-maximum suppression.
    """
    if len(proposals) == 0:
        return []
    
    # Sort by confidence
    proposals = sorted(proposals, key=lambda p: p.confidence, reverse=True)
    
    kept = []
    
    for prop in proposals:
        x1, y1, w1, h1 = prop.bbox
        
        # Check overlap with kept proposals
        suppress = False
        for kept_prop in kept:
            x2, y2, w2, h2 = kept_prop.bbox
            
            # Calculate IOU
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xb - xa) * max(0, yb - ya)
            
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            if iou > iou_threshold:
                suppress = True
                break
        
        if not suppress:
            kept.append(prop)
    
    return kept


def propose_regions_cv_fast(
    image: np.ndarray,
    bbox_scale_factor: float = 3.0,
    min_area: int = 1000,
    max_area: int = 100000,
    min_eccentricity: float = 0.60,
    background_kernel: int = 51,
    threshold_method: str = 'adaptive',
    iou_threshold: float = 0.3,
    detect_plate: bool = True,
    plate_shrink: int = 50,
    verbose: bool = False
) -> Tuple[List[RegionProposal], Optional[Tuple[int, int, int]]]:
    """
    Fast CV-based region proposal with organism-scale bboxes.
    
    Main entry point for fast CV proposals.
    
    Args:
        image: Input RGB image
        bbox_scale_factor: How much to expand bboxes (default 3.0 = 3x)
        min_area: Minimum organism bbox area (px)
        max_area: Maximum organism bbox area (px)
        min_eccentricity: Minimum eccentricity (elongation)
        background_kernel: Kernel size for background subtraction
        threshold_method: 'otsu', 'adaptive', or 'fixed'
        iou_threshold: NMS IOU threshold for merging
        detect_plate: If True, automatically detect and mask circular plate
        plate_shrink: Shrink plate mask by this many pixels to avoid edges
        verbose: Print progress
        
    Returns:
        Tuple of (proposals, detected_circle)
        - proposals: List of RegionProposal objects
        - detected_circle: (cx, cy, radius) if plate detected, None otherwise
    """
    if verbose:
        print(f"[CV Fast] Input shape: {image.shape}")
    
    # Step 0: Detect plate (but don't mask image yet)
    detected_circle = None
    plate_mask = None
    if detect_plate:
        if verbose:
            print("[CV Fast] Detecting circular plate...")
        from .detect_plate import detect_circular_plate, create_plate_mask
        detected_circle = detect_circular_plate(image, verbose=verbose)
        if detected_circle is not None:
            plate_mask = create_plate_mask(
                image.shape[:2],
                detected_circle,
                shrink_radius=plate_shrink,
                feather=0  # No feathering for binary operations
            )
            if verbose:
                print(f"[CV Fast] Plate detected: center=({detected_circle[0]}, {detected_circle[1]}), radius={detected_circle[2]}")
        else:
            if verbose:
                print("[CV Fast] Plate detection failed, using full image")
    
    # Step 1: Preprocess
    if verbose:
        print("[CV Fast] Enhancing contrast...")
    enhanced = preprocess_image(image)
    
    # Step 2: Background subtraction
    if verbose:
        print(f"[CV Fast] Subtracting background (kernel={background_kernel})...")
    fg = subtract_background(enhanced, kernel_size=background_kernel, morphology_op='tophat')
    
    # Step 3: Threshold
    if verbose:
        print(f"[CV Fast] Thresholding ({threshold_method})...")
    binary = threshold_image(fg, method=threshold_method)
    
    # Step 4: Apply plate mask to binary (not to input image!)
    if plate_mask is not None:
        if verbose:
            print("[CV Fast] Applying plate mask to foreground...")
        binary = cv2.bitwise_and(binary, plate_mask)
    
    # Step 5: Morphological cleanup
    if verbose:
        print("[CV Fast] Morphological cleanup...")
    from skimage import morphology
    binary = morphology.remove_small_objects(binary > 0, min_size=50)
    binary = (binary * 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 6: Find contours
    if verbose:
        print("[CV Fast] Finding contours...")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 7: Expand bboxes and merge
    if verbose:
        print(f"[CV Fast] Expanding bboxes (scale={bbox_scale_factor}) and merging...")
    proposals = expand_and_merge_bboxes(
        contours,
        image.shape[:2],
        bbox_scale_factor=bbox_scale_factor,
        min_area=min_area,
        max_area=max_area,
        min_eccentricity=min_eccentricity,
        iou_threshold=iou_threshold,
        verbose=verbose
    )
    
    if verbose:
        print(f"[CV Fast] Final proposals: {len(proposals)}")
    
    return proposals, detected_circle
