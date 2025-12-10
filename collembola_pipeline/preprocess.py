"""Image preprocessing utilities for collembola detection."""

from __future__ import annotations
import numpy as np
from typing import Optional


def normalize_brightness(
    image: np.ndarray,
    target_median: float = 50.0,
    auto: bool = True,
    threshold: float = 45.0,
) -> np.ndarray:
    """
    Normalize image brightness to improve detection on dark/light images.

    Args:
        image: RGB image as numpy array (H, W, 3)
        target_median: Target median brightness value (0-255)
        auto: If True, only normalize if current median differs significantly
        threshold: Minimum median brightness threshold for auto normalization

    Returns:
        Brightness-normalized image as uint8 array
    """
    current_median = float(np.median(image))

    # Auto mode: only normalize if image is significantly dark
    if auto and current_median >= threshold:
        return image

    # Calculate adjustment needed
    adjustment = target_median - current_median

    # Apply adjustment and clip to valid range
    adjusted = np.clip(image.astype(float) + adjustment, 0, 255).astype(np.uint8)

    return adjusted


def get_brightness_stats(image: np.ndarray) -> dict:
    """Get brightness statistics for an image."""
    return {
        "median": float(np.median(image)),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean_rgb": tuple(float(image[:, :, i].mean()) for i in range(3)),
    }
