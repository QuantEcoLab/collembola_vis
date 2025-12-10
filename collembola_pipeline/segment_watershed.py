"""
Watershed-based segmentation refinement for collembola detection.

Refines bounding box proposals to precise pixel-level masks using
marker-based watershed segmentation.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import morphology, filters
from skimage.segmentation import watershed


def refine_mask_watershed(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: int = 20,
    use_grabcut: bool = False
) -> np.ndarray:
    """
    Refine a bounding box to precise mask using watershed segmentation.
    
    Args:
        image: RGB image (H, W, 3)
        bbox: (x, y, w, h) bounding box
        margin: Pixels to add around bbox for context
        use_grabcut: Use GrabCut instead of watershed (slower but better)
        
    Returns:
        Binary mask (H, W) of refined segmentation
    """
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # Extract region with margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    
    crop = image[y1:y2, x1:x2].copy()
    
    if use_grabcut:
        # GrabCut approach (slower but more accurate)
        mask_crop = _grabcut_segment(crop, bbox, margin)
    else:
        # Watershed approach (faster)
        mask_crop = _watershed_segment(crop)
    
    # Create full-size mask
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask[y1:y2, x1:x2] = mask_crop
    
    return mask


def _watershed_segment(crop: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation on crop.
    
    Args:
        crop: RGB crop image
        
    Returns:
        Binary mask of crop
    """
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.copy()
    
    # Denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge-preserving bilateral filter
    filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    # Otsu threshold to get rough foreground
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find sure foreground (high distance from background)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Find sure background (dilate binary)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background is 1, unknown is 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    # Convert to 3-channel for watershed
    if len(crop.shape) == 2:
        crop_3ch = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_3ch = crop.copy()
    
    markers = cv2.watershed(crop_3ch, markers)
    
    # Create mask (watershed boundary is -1, background is 1, foreground is >1)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[markers > 1] = 255
    
    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Keep only the largest component (the organism)
    mask = _keep_largest_component(mask)
    
    return mask


def _grabcut_segment(crop: np.ndarray, bbox: Tuple[int, int, int, int], margin: int) -> np.ndarray:
    """
    GrabCut segmentation on crop.
    
    Args:
        crop: RGB crop image
        bbox: Original bbox (x, y, w, h)
        margin: Margin used for cropping
        
    Returns:
        Binary mask of crop
    """
    # Convert to RGB if grayscale
    if len(crop.shape) == 2:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = crop.copy()
    
    # Initialize mask
    mask = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    
    # Define rectangle (bbox within crop coordinates)
    x, y, w, h = bbox
    rect_x = margin
    rect_y = margin
    rect_w = w
    rect_h = h
    
    # Ensure rect is within crop bounds
    crop_h, crop_w = crop_rgb.shape[:2]
    rect_x = max(0, min(rect_x, crop_w - 1))
    rect_y = max(0, min(rect_y, crop_h - 1))
    rect_w = min(rect_w, crop_w - rect_x)
    rect_h = min(rect_h, crop_h - rect_y)
    
    if rect_w < 5 or rect_h < 5:
        # Too small for GrabCut, fallback to simple threshold
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    rect = (rect_x, rect_y, rect_w, rect_h)
    
    # GrabCut
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    try:
        cv2.grabCut(crop_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255
    except:
        # GrabCut failed, fallback to threshold
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _keep_largest_component(mask)
    
    return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    # Find largest component (excluding background which is label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only largest component
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    
    return result


def refine_proposals_watershed(
    image: np.ndarray,
    proposals: list,
    use_grabcut: bool = False,
    verbose: bool = False
) -> list:
    """
    Refine all proposals with watershed segmentation.
    
    Args:
        image: RGB image (H, W, 3)
        proposals: List of RegionProposal objects
        use_grabcut: Use GrabCut instead of watershed
        verbose: Print progress
        
    Returns:
        Updated proposals with refined masks
    """
    if verbose:
        print(f"[Watershed] Refining {len(proposals)} masks...")
    
    for i, prop in enumerate(proposals):
        refined_mask = refine_mask_watershed(image, prop.bbox, use_grabcut=use_grabcut)
        prop.mask = refined_mask
        
        if verbose and (i + 1) % 100 == 0:
            print(f"[Watershed] Refined {i + 1}/{len(proposals)} masks")
    
    if verbose:
        print(f"[Watershed] Done refining {len(proposals)} masks")
    
    return proposals
