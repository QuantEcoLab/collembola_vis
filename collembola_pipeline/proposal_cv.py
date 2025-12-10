"""
CV-based region proposal for collembola detection.

Fast alternative to SAM using classical computer vision:
- Background subtraction via morphological opening
- Contrast enhancement with CLAHE
- Adaptive thresholding
- Contour detection and filtering

Expected speedup: 10x faster than SAM (30s vs 5min per plate)
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import cv2
from skimage import morphology, measure
from dataclasses import dataclass


@dataclass
class RegionProposal:
    """A proposed region that might contain an organism"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mask: np.ndarray  # Binary mask (H, W)
    confidence: float  # Proposal confidence score
    area: int  # Pixel area
    eccentricity: float  # Shape eccentricity


def preprocess_image(
    image: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8
) -> np.ndarray:
    """
    Preprocess plate image for better contrast.
    
    Args:
        image: RGB image (H, W, 3)
        clahe_clip_limit: CLAHE clipping limit
        clahe_grid_size: CLAHE grid size
        
    Returns:
        Enhanced grayscale image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_grid_size, clahe_grid_size)
    )
    enhanced = clahe.apply(gray)
    
    return enhanced


def subtract_background(
    gray: np.ndarray,
    kernel_size: int = 51,
    morphology_op: str = 'opening'
) -> np.ndarray:
    """
    Remove background using morphological operation.
    
    Args:
        gray: Grayscale image
        kernel_size: Size of morphological kernel (larger = more background removal)
        morphology_op: 'opening', 'tophat', or 'blackhat'
        
    Returns:
        Background-subtracted image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if morphology_op == 'opening':
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        # Subtract background
        result = cv2.subtract(gray, background)
    elif morphology_op == 'tophat':
        # Top-hat: original - opening (bright objects on dark background)
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif morphology_op == 'blackhat':
        # Black-hat: closing - original (dark objects on bright background)
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError(f"Unknown morphology operation: {morphology_op}")
    
    return result


def threshold_image(
    image: np.ndarray,
    method: str = 'adaptive',
    block_size: int = 51,
    c: float = 2
) -> np.ndarray:
    """
    Threshold image to binary.
    
    Args:
        image: Grayscale image
        method: 'otsu', 'adaptive', or 'fixed'
        block_size: Block size for adaptive threshold
        c: Constant subtracted from mean (adaptive)
        
    Returns:
        Binary mask
    """
    if method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
    elif method == 'fixed':
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return binary


def extract_contours(
    binary: np.ndarray,
    min_area: int = 200,
    max_area: int = 20000,
    min_eccentricity: float = 0.70,
    max_eccentricity: float = 0.999,
    min_solidity: float = 0.50
) -> List[RegionProposal]:
    """
    Extract and filter contours from binary image.
    
    Args:
        binary: Binary mask
        min_area: Minimum contour area in pixels
        max_area: Maximum contour area in pixels
        min_eccentricity: Minimum eccentricity (0=circle, 1=line)
        max_eccentricity: Maximum eccentricity
        min_solidity: Minimum solidity (convex hull ratio)
        
    Returns:
        List of region proposals
    """
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate shape properties
        if len(contour) >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    eccentricity = 0.0
            except:
                eccentricity = 0.5  # Default if fit fails
        else:
            eccentricity = 0.5
        
        # Filter by eccentricity (organisms are elongated)
        if eccentricity < min_eccentricity or eccentricity > max_eccentricity:
            continue
        
        # Calculate solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Filter by solidity
        if solidity < min_solidity:
            continue
        
        # Create mask for this contour
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Confidence score (higher for better shape match)
        confidence = (eccentricity * 0.5) + (solidity * 0.5)
        
        proposals.append(RegionProposal(
            bbox=(x, y, w, h),
            mask=mask,
            confidence=confidence,
            area=int(area),
            eccentricity=eccentricity
        ))
    
    # Sort by confidence (best proposals first)
    proposals.sort(key=lambda p: p.confidence, reverse=True)
    
    return proposals


def propose_regions_cv(
    image: np.ndarray,
    min_area: int = 200,
    max_area: int = 20000,
    min_eccentricity: float = 0.70,
    background_kernel: int = 51,
    threshold_method: str = 'adaptive',
    verbose: bool = False
) -> List[RegionProposal]:
    """
    Main CV-based region proposal function.
    
    Args:
        image: Input RGB or grayscale image
        min_area: Minimum organism area in pixels
        max_area: Maximum organism area in pixels
        min_eccentricity: Minimum eccentricity (elongation)
        background_kernel: Kernel size for background subtraction
        threshold_method: 'otsu', 'adaptive', or 'fixed'
        verbose: Print progress messages
        
    Returns:
        List of region proposals sorted by confidence
    """
    if verbose:
        print(f"[CV Proposal] Input shape: {image.shape}")
    
    # Step 1: Preprocess (CLAHE contrast enhancement)
    if verbose:
        print("[CV Proposal] Enhancing contrast...")
    enhanced = preprocess_image(image)
    
    # Step 2: Background subtraction
    if verbose:
        print(f"[CV Proposal] Subtracting background (kernel={background_kernel})...")
    fg = subtract_background(enhanced, kernel_size=background_kernel, morphology_op='tophat')
    
    # Step 3: Threshold to binary
    if verbose:
        print(f"[CV Proposal] Thresholding ({threshold_method})...")
    binary = threshold_image(fg, method=threshold_method)
    
    # Step 4: Morphological cleanup
    if verbose:
        print("[CV Proposal] Morphological cleanup...")
    # Remove small noise
    binary = morphology.remove_small_objects(binary > 0, min_size=min_area // 4)
    binary = (binary * 255).astype(np.uint8)
    
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Extract and filter contours
    if verbose:
        print("[CV Proposal] Extracting contours...")
    proposals = extract_contours(
        binary,
        min_area=min_area,
        max_area=max_area,
        min_eccentricity=min_eccentricity
    )
    
    if verbose:
        print(f"[CV Proposal] Found {len(proposals)} region proposals")
    
    return proposals


# Convenience function to get just bboxes (for compatibility with existing code)
def propose_bboxes_cv(image: np.ndarray, **kwargs) -> List[Tuple[int, int, int, int]]:
    """Get just bounding boxes (no masks)"""
    proposals = propose_regions_cv(image, **kwargs)
    return [p.bbox for p in proposals]
