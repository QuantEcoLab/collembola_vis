"""
Blob-based region proposal for collembola detection.

New approach: Use blob detection (Difference of Gaussian / Laplacian of Gaussian)
which is specifically designed to detect blob-like structures (like organisms)
and NOT random texture/edges.

Organisms are blob-like structures with:
- Sizes ranging from ~35px to ~250px diameter
- Circular to elliptical shape
- Distinct from background (either darker or brighter)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2
from skimage.feature import blob_dog, blob_log
from skimage.color import rgb2gray

from .proposal_cv import RegionProposal
from .detect_plate import detect_circular_plate, create_plate_mask


def propose_regions_blob(
    image: np.ndarray,
    min_sigma: float = 5,
    max_sigma: float = 100,
    num_sigma: int = 20,
    threshold: float = 0.01,
    overlap: float = 0.5,
    bbox_expand: float = 2.5,
    min_area: int = 1500,
    max_area: int = 100000,
    detect_plate: bool = True,
    plate_shrink: int = 50,
    verbose: bool = False
) -> Tuple[List[RegionProposal], Optional[Tuple[int, int, int]]]:
    """
    Generate region proposals using blob detection (LoG).
    
    This method specifically detects blob-like structures, making it much more
    selective than texture/edge detection.
    
    Args:
        image: Input RGB image
        min_sigma: Minimum blob size (pixels radius)
        max_sigma: Maximum blob size (pixels radius)
        num_sigma: Number of scales to search
        threshold: Blob detection threshold (lower = more blobs)
        overlap: Maximum overlap between blobs before merging
        bbox_expand: Expand bboxes by this factor
        min_area: Minimum proposal area
        max_area: Maximum proposal area
        detect_plate: Auto-detect and mask plate
        plate_shrink: Shrink plate mask
        verbose: Print progress
        
    Returns:
        (proposals, detected_circle)
    """
    if verbose:
        print(f"[Blob Detection] Input shape: {image.shape}")
    
    H, W = image.shape[:2]
    
    # Step 1: Detect plate
    detected_circle = None
    plate_mask = None
    if detect_plate:
        if verbose:
            print("[Blob Detection] Detecting plate...")
        detected_circle = detect_circular_plate(image, verbose=verbose)
        if detected_circle:
            plate_mask = create_plate_mask((H, W), detected_circle, shrink_radius=plate_shrink)
    
    # Step 2: Convert to grayscale
    gray = rgb2gray(image)
    
    # Apply plate mask if available
    if plate_mask is not None:
        gray_masked = gray.copy()
        gray_masked[plate_mask == 0] = gray.mean()  # Set outside to mean gray
        gray = gray_masked
    
    # Step 3: Blob detection using Laplacian of Gaussian
    if verbose:
        print(f"[Blob Detection] Running LoG blob detection (sigma={min_sigma}-{max_sigma})...")
    
    blobs = blob_log(
        gray,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap
    )
    
    # blobs shape: (n, 3) where columns are (y, x, sigma)
    # radius = sigma * sqrt(2)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    if verbose:
        print(f"[Blob Detection] Found {len(blobs)} blobs")
    
    # Step 4: Convert blobs to proposals
    proposals = []
    
    for blob in blobs:
        y, x, r = blob
        x, y, r = int(x), int(y), int(r)
        
        # Create bbox from blob circle
        bbox_r = int(r * bbox_expand)
        
        x0 = max(0, x - bbox_r)
        y0 = max(0, y - bbox_r)
        x1 = min(W, x + bbox_r)
        y1 = min(H, y + bbox_r)
        
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        area = bbox_w * bbox_h
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        
        # Confidence based on blob response strength
        confidence = min(1.0, r / max_sigma)  # Larger blobs = higher confidence
        
        proposals.append(RegionProposal(
            bbox=(x0, y0, bbox_w, bbox_h),
            mask=mask,
            confidence=confidence,
            area=area,
            eccentricity=0.0
        ))
    
    if verbose:
        print(f"[Blob Detection] After area filtering: {len(proposals)} proposals")
    
    # Step 5: Sort by confidence
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    
    return proposals, detected_circle
