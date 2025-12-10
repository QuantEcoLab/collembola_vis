"""
SAM-based region proposal for collembola detection.

Uses Segment Anything Model to generate high-quality mask proposals.
Converts SAM masks to RegionProposal objects compatible with the pipeline.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from .segment import generate_masks
from .proposal_cv import RegionProposal


def sam_mask_to_proposal(sam_mask: Dict[str, Any]) -> RegionProposal:
    """
    Convert a SAM mask dictionary to a RegionProposal.
    
    Args:
        sam_mask: Dictionary with 'bbox', 'segmentation', 'area', etc.
        
    Returns:
        RegionProposal object
    """
    bbox = sam_mask.get('bbox', [0, 0, 0, 0])
    x, y, w, h = bbox
    
    # Get mask (binary array)
    seg = sam_mask.get('segmentation')
    if seg is None:
        # Create mask from bbox if no segmentation
        H, W = 1000, 1000  # Placeholder, should be image size
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
    else:
        mask = (seg.astype(np.uint8) * 255)
    
    # Get area
    area = sam_mask.get('area', int(mask.sum() / 255))
    
    # Calculate eccentricity from mask
    from skimage.measure import regionprops
    props = regionprops(mask > 0)
    if props:
        eccentricity = float(props[0].eccentricity)
    else:
        eccentricity = 0.5
    
    # Use predicted_iou as confidence (SAM provides this)
    confidence = float(sam_mask.get('predicted_iou', 0.9))
    
    return RegionProposal(
        bbox=(x, y, w, h),
        mask=mask,
        confidence=confidence,
        area=area,
        eccentricity=eccentricity
    )


def propose_regions_sam(
    image: np.ndarray,
    device: str = 'cuda',
    min_area: int = 200,
    max_area: int = 100000,
    verbose: bool = False
) -> List[RegionProposal]:
    """
    Generate region proposals using SAM.
    
    Args:
        image: RGB image (H, W, 3)
        device: 'cuda' or 'cpu'
        min_area: Minimum mask area in pixels
        max_area: Maximum mask area in pixels
        verbose: Print progress messages
        
    Returns:
        List of RegionProposal objects
    """
    if verbose:
        print(f"[SAM Proposal] Generating masks on {device}...")
    
    # Generate SAM masks
    sam_masks = generate_masks(image, device=device)
    
    if verbose:
        print(f"[SAM Proposal] Generated {len(sam_masks)} raw masks")
    
    # Convert to RegionProposal objects and filter by area
    proposals = []
    for sam_mask in sam_masks:
        area = sam_mask.get('area', 0)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        proposal = sam_mask_to_proposal(sam_mask)
        proposals.append(proposal)
    
    # Sort by confidence (best proposals first)
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    
    if verbose:
        print(f"[SAM Proposal] Filtered to {len(proposals)} proposals (area: {min_area}-{max_area})")
    
    return proposals


def propose_regions_sam_fast(
    image: np.ndarray,
    device: str = 'cuda',
    min_area: int = 200,
    max_area: int = 100000,
    verbose: bool = False
) -> List[RegionProposal]:
    """
    Fast SAM proposal - returns masks directly without converting to RegionProposal.
    Just stores the SAM mask dict in the proposal for later use.
    
    This avoids redundant regionprops computation.
    """
    if verbose:
        print(f"[SAM Proposal] Generating masks on {device}...")
    
    sam_masks = generate_masks(image, device=device)
    
    if verbose:
        print(f"[SAM Proposal] Generated {len(sam_masks)} raw masks")
    
    # Filter and convert
    proposals = []
    for sam_mask in sam_masks:
        area = sam_mask.get('area', 0)
        
        if area < min_area or area > max_area:
            continue
        
        bbox = sam_mask.get('bbox', [0, 0, 0, 0])
        x, y, w, h = bbox
        
        seg = sam_mask.get('segmentation')
        if seg is not None:
            mask = (seg.astype(np.uint8) * 255)
        else:
            # Create bbox mask
            H, W = image.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
        
        # Quick eccentricity estimate (avoid full regionprops)
        # Use predicted_iou as confidence
        confidence = float(sam_mask.get('predicted_iou', 0.9))
        
        # Rough eccentricity from bbox aspect ratio
        if h > 0:
            ar = w / h
            if ar > 1:
                ar = 1 / ar
            # Map aspect ratio to eccentricity (0=circle, 1=line)
            eccentricity = np.sqrt(1 - ar**2)
        else:
            eccentricity = 0.5
        
        proposal = RegionProposal(
            bbox=(x, y, w, h),
            mask=mask,
            confidence=confidence,
            area=area,
            eccentricity=eccentricity
        )
        proposals.append(proposal)
    
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    
    if verbose:
        print(f"[SAM Proposal] Filtered to {len(proposals)} proposals")
    
    return proposals
