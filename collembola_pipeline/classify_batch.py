"""
Batch CNN classification for region proposals.

Processes multiple proposals in parallel for GPU efficiency.
Uses trained ResNet18 classifier with 224x224 input.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .classify import load_classifier, get_val_transform
from .config import CLASSIFIER_THRESHOLD
from .proposal_cv import RegionProposal


def crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], margin: int = 10) -> Image.Image:
    """
    Crop image region from bounding box with margin.
    
    Args:
        image: RGB image (H, W, 3)
        bbox: (x, y, w, h)
        margin: Pixels to add around bbox
        
    Returns:
        PIL Image crop
    """
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # Add margin and clip to image bounds
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    
    crop = image[y1:y2, x1:x2]
    
    # Convert to PIL Image
    if len(crop.shape) == 2:
        crop = np.stack([crop] * 3, axis=-1)
    
    return Image.fromarray(crop)


def classify_proposals_batch(
    image: np.ndarray,
    proposals: List[RegionProposal],
    model=None,
    threshold: float = CLASSIFIER_THRESHOLD,
    batch_size: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = False
) -> Tuple[List[int], List[float]]:
    """
    Classify region proposals in batches.
    
    Args:
        image: RGB image (H, W, 3)
        proposals: List of RegionProposal objects
        model: Trained classifier (if None, will load from config)
        threshold: Classification threshold
        batch_size: Number of crops per batch
        device: 'cuda' or 'cpu'
        verbose: Print progress
        
    Returns:
        accepted_indices: Indices of proposals classified as organisms
        confidences: Classification confidence for all proposals
    """
    if model is None:
        if verbose:
            print(f"[Batch Classify] Loading model to {device}...")
        model = load_classifier(device)
    
    if len(proposals) == 0:
        return [], []
    
    # Prepare transform
    transform = get_val_transform(img_size=224)
    
    # Process in batches
    all_confidences = []
    num_batches = (len(proposals) + batch_size - 1) // batch_size
    
    if verbose:
        print(f"[Batch Classify] Processing {len(proposals)} proposals in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(proposals))
        batch_proposals = proposals[start_idx:end_idx]
        
        # Crop and transform all images in batch
        batch_tensors = []
        for prop in batch_proposals:
            crop_img = crop_bbox(image, prop.bbox, margin=10)
            tensor = transform(crop_img)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            confidences = probs[:, 1].cpu().numpy()
        
        all_confidences.extend(confidences.tolist())
        
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"[Batch Classify] Batch {batch_idx + 1}/{num_batches} done")
    
    # Find accepted indices
    accepted_indices = [i for i, conf in enumerate(all_confidences) if conf >= threshold]
    
    if verbose:
        print(f"[Batch Classify] Accepted {len(accepted_indices)}/{len(proposals)} proposals (threshold={threshold:.2f})")
    
    return accepted_indices, all_confidences


def filter_proposals_by_classification(
    image: np.ndarray,
    proposals: List[RegionProposal],
    model=None,
    threshold: float = CLASSIFIER_THRESHOLD,
    batch_size: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = False
) -> Tuple[List[RegionProposal], List[float]]:
    """
    Filter proposals by CNN classification, returning accepted proposals.
    
    Args:
        image: RGB image (H, W, 3)
        proposals: List of RegionProposal objects
        model: Trained classifier (if None, will load from config)
        threshold: Classification threshold
        batch_size: Number of crops per batch
        device: 'cuda' or 'cpu'
        verbose: Print progress
        
    Returns:
        accepted_proposals: Proposals classified as organisms
        accepted_confidences: Their classification confidences
    """
    accepted_indices, all_confidences = classify_proposals_batch(
        image, proposals, model, threshold, batch_size, device, verbose
    )
    
    accepted_proposals = [proposals[i] for i in accepted_indices]
    accepted_confidences = [all_confidences[i] for i in accepted_indices]
    
    return accepted_proposals, accepted_confidences
