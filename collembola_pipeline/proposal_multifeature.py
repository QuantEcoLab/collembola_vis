"""
Multi-feature CV-based region proposal for collembola detection.

New approach based on organism analysis:
- 56% of organisms have low contrast (<15) - can't use simple thresholding
- 78% have high texture (std > 20) - USE TEXTURE!
- Organisms have higher local variance than background

Strategy:
1. Detect plate and mask outside
2. Multi-scale texture detection (local std, Sobel edges)
3. Local variance/entropy to find "interesting" regions
4. Combine features into saliency map
5. Find connected components in saliency map
6. Generate proposals from components
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage

from .proposal_cv import RegionProposal
from .detect_plate import detect_circular_plate, create_plate_mask


def compute_texture_map(
    image: np.ndarray,
    kernel_size: int = 15,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute local texture (standard deviation) map.
    
    Organisms have higher local texture than smooth background.
    """
    if verbose:
        print(f"[Texture] Computing local std with kernel={kernel_size}...")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    # Local mean and std
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq_mean = cv2.filter2D(gray ** 2, -1, kernel)
    local_std = np.sqrt(np.maximum(0, local_sq_mean - local_mean ** 2))
    
    return local_std


def compute_edge_map(
    image: np.ndarray,
    low_thresh: int = 30,
    high_thresh: int = 100,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute edge map using Canny edge detection.
    """
    if verbose:
        print(f"[Edges] Computing Canny edges...")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # Dilate edges slightly to create regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


def compute_local_variance_map(
    image: np.ndarray,
    kernel_size: int = 21,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute local variance using integral images (fast).
    """
    if verbose:
        print(f"[Variance] Computing local variance with kernel={kernel_size}...")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    
    # Use box filter for fast local variance
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(kernel_size, kernel_size))
    sq_mean = cv2.boxFilter(gray ** 2, ddepth=-1, ksize=(kernel_size, kernel_size))
    variance = sq_mean - mean ** 2
    
    return variance.astype(np.float32)


def create_saliency_map(
    texture_map: np.ndarray,
    edge_map: np.ndarray,
    variance_map: np.ndarray,
    texture_weight: float = 0.5,
    edge_weight: float = 0.3,
    variance_weight: float = 0.2,
    verbose: bool = False
) -> np.ndarray:
    """
    Combine multiple feature maps into a single saliency map.
    """
    if verbose:
        print("[Saliency] Combining feature maps...")
    
    # Normalize each map to [0, 1]
    def normalize(x):
        x = x.astype(np.float32)
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    
    texture_norm = normalize(texture_map)
    edge_norm = normalize(edge_map)
    variance_norm = normalize(variance_map)
    
    # Weighted combination
    saliency = (
        texture_weight * texture_norm +
        edge_weight * edge_norm +
        variance_weight * variance_norm
    )
    
    # Normalize to [0, 255]
    saliency = (normalize(saliency) * 255).astype(np.uint8)
    
    return saliency


def propose_regions_multifeature(
    image: np.ndarray,
    texture_kernel: int = 15,
    variance_kernel: int = 21,
    saliency_threshold: int = 40,  # Threshold on combined saliency (0-255)
    min_area: int = 2500,  # Increased - 90% of GT organisms are >2853 px²
    max_area: int = 100000,
    bbox_expand: float = 1.5,  # Expand bboxes by this factor
    min_solidity: float = 0.3,  # NEW: Filter out non-blob-like shapes
    min_extent: float = 0.2,  # NEW: bbox fill ratio
    max_aspect_ratio: float = 5.0,  # NEW: Filter extreme elongations
    iou_threshold: float = 0.3,
    detect_plate: bool = True,
    plate_shrink: int = 50,
    verbose: bool = False
) -> Tuple[List[RegionProposal], Optional[Tuple[int, int, int]]]:
    """
    Generate region proposals using multiple visual features.
    
    This method doesn't rely on intensity alone, making it robust to low-contrast organisms.
    Includes shape filters to reject noise (dots, lines, irregular shapes).
    
    Args:
        image: Input RGB image
        texture_kernel: Kernel size for texture detection
        variance_kernel: Kernel size for variance detection
        saliency_threshold: Threshold for saliency map (0-255)
        min_area: Minimum proposal area (default 2500 to filter small noise)
        max_area: Maximum proposal area
        bbox_expand: Expand bboxes by this factor
        min_solidity: Minimum solidity (convex hull ratio) - filters irregular shapes
        min_extent: Minimum extent (contour area / bbox area) - filters sparse regions
        max_aspect_ratio: Maximum aspect ratio - filters extreme elongations
        iou_threshold: NMS threshold
        detect_plate: Auto-detect and mask plate
        plate_shrink: Shrink plate mask by this many pixels
        verbose: Print progress
        
    Returns:
        (proposals, detected_circle)
    """
    if verbose:
        print(f"[Multi-Feature] Input shape: {image.shape}")
    
    H, W = image.shape[:2]
    
    # Step 1: Detect plate
    detected_circle = None
    plate_mask = None
    if detect_plate:
        if verbose:
            print("[Multi-Feature] Detecting plate...")
        detected_circle = detect_circular_plate(image, verbose=verbose)
        if detected_circle:
            plate_mask = create_plate_mask((H, W), detected_circle, shrink_radius=plate_shrink)
    
    # Step 2: Compute feature maps
    texture_map = compute_texture_map(image, kernel_size=texture_kernel, verbose=verbose)
    edge_map = compute_edge_map(image, verbose=verbose)
    variance_map = compute_local_variance_map(image, kernel_size=variance_kernel, verbose=verbose)
    
    # Step 3: Create saliency map
    saliency = create_saliency_map(
        texture_map,
        edge_map,
        variance_map,
        texture_weight=0.5,  # Texture is most important (78% of organisms have high texture)
        edge_weight=0.3,
        variance_weight=0.2,
        verbose=verbose
    )
    
    # Step 4: Apply plate mask to saliency
    if plate_mask is not None:
        if verbose:
            print("[Multi-Feature] Applying plate mask...")
        saliency = cv2.bitwise_and(saliency, plate_mask)
    
    # Step 5: Threshold saliency map
    if verbose:
        print(f"[Multi-Feature] Thresholding saliency at {saliency_threshold}...")
    _, binary = cv2.threshold(saliency, saliency_threshold, 255, cv2.THRESH_BINARY)
    
    # Step 6: Morphological cleanup
    if verbose:
        print("[Multi-Feature] Morphological cleanup...")
    
    # Remove small noise
    from skimage import morphology
    binary = morphology.remove_small_objects(binary > 0, min_size=100)
    binary = (binary * 255).astype(np.uint8)
    
    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 7: Find connected components
    if verbose:
        print("[Multi-Feature] Finding connected components...")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if verbose:
        print(f"[Multi-Feature] Found {len(contours)} initial regions")
    
    # Step 8: Create proposals from contours with shape filtering
    proposals = []
    rejected_shape = 0
    
    for contour in contours:
        # Calculate shape properties on original contour (before expansion)
        contour_area = cv2.contourArea(contour)
        if contour_area < 10:  # Skip tiny contours
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        # Shape filter 1: Extent (how much of bbox is filled)
        extent = contour_area / bbox_area if bbox_area > 0 else 0
        if extent < min_extent:
            rejected_shape += 1
            continue
        
        # Shape filter 2: Aspect ratio
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        if aspect_ratio > max_aspect_ratio:
            rejected_shape += 1
            continue
        
        # Shape filter 3: Solidity (contour area vs convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            rejected_shape += 1
            continue
        
        # Expand bbox
        cx, cy = x + w / 2, y + h / 2
        new_w = w * bbox_expand
        new_h = h * bbox_expand
        
        new_x = int(cx - new_w / 2)
        new_y = int(cy - new_h / 2)
        new_w = int(new_w)
        new_h = int(new_h)
        
        # Clip to image
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(W - new_x, new_w)
        new_h = min(H - new_y, new_h)
        
        expanded_area = new_w * new_h
        
        # Filter by expanded area
        if expanded_area < min_area or expanded_area > max_area:
            continue
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
        
        # Calculate confidence (mean saliency in region)
        roi_saliency = saliency[new_y:new_y+new_h, new_x:new_x+new_w]
        confidence = roi_saliency.mean() / 255.0
        
        proposals.append(RegionProposal(
            bbox=(new_x, new_y, new_w, new_h),
            mask=mask,
            confidence=confidence,
            area=expanded_area,
            eccentricity=0.0  # Not computed
        ))
    
    if verbose:
        print(f"[Multi-Feature] Rejected {rejected_shape} regions due to shape filters")
        print(f"[Multi-Feature] After area+shape filtering: {len(proposals)} proposals")
    
    # Step 9: Non-maximum suppression
    if verbose:
        print("[Multi-Feature] Applying NMS...")
    proposals = non_max_suppression(proposals, iou_threshold, verbose=verbose)
    
    if verbose:
        print(f"[Multi-Feature] Final proposals: {len(proposals)}")
    
    return proposals, detected_circle


def non_max_suppression(
    proposals: List[RegionProposal],
    iou_threshold: float = 0.3,
    verbose: bool = False
) -> List[RegionProposal]:
    """NMS to merge overlapping proposals."""
    if len(proposals) == 0:
        return []
    
    # Sort by confidence
    proposals = sorted(proposals, key=lambda p: p.confidence, reverse=True)
    
    kept = []
    for prop in proposals:
        x1, y1, w1, h1 = prop.bbox
        
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
