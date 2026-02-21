"""Calibration services — auto ruler detection and manual fallback."""

import json
import math
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.config import settings


def auto_calibrate(image_path: Path, ruler_mm: float = 10.0) -> dict:
    """Attempt automatic ruler calibration using CV.

    1. Convert to grayscale, Gaussian blur
    2. Canny edge detection
    3. Hough line transform to find ruler lines
    4. Detect periodic tick spacing
    5. Compute um_per_pixel from known mm between ticks

    Returns dict with um_per_pixel, confidence, method, etc.
    """
    Image.MAX_IMAGE_PIXELS = None
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Focus on edges of image where rulers typically are (bottom 15%)
    h, w = gray.shape
    roi_top = int(h * 0.85)
    roi = gray[roi_top:, :]

    # Preprocessing
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough line detection — look for vertical tick marks
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=15, maxLineGap=10
    )

    if lines is None or len(lines) < 3:
        return {
            "um_per_pixel": None,
            "confidence": 0.0,
            "method": "auto_failed",
            "error": "Could not detect enough ruler lines",
        }

    # Extract x-positions of roughly vertical lines
    vertical_xs = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1))
        # Near-vertical: angle close to pi/2
        if angle > math.pi / 4:
            vertical_xs.append((x1 + x2) / 2)

    if len(vertical_xs) < 3:
        return {
            "um_per_pixel": None,
            "confidence": 0.0,
            "method": "auto_failed",
            "error": "Not enough vertical tick marks detected",
        }

    # Sort and find dominant spacing via histogram of pairwise distances
    vertical_xs = sorted(set(int(x) for x in vertical_xs))
    spacings = [vertical_xs[i + 1] - vertical_xs[i] for i in range(len(vertical_xs) - 1)]
    spacings = [s for s in spacings if s > 10]  # filter noise

    if not spacings:
        return {
            "um_per_pixel": None,
            "confidence": 0.0,
            "method": "auto_failed",
            "error": "Could not determine tick spacing",
        }

    # Use median spacing as the dominant tick interval
    median_spacing = float(np.median(spacings))

    # Assume ruler_mm is the total length and we count tick intervals
    # Estimate number of intervals: total ruler px / median spacing
    total_ruler_px = median_spacing * round(ruler_mm)  # assume 1mm per tick
    ruler_um = ruler_mm * 1000.0
    um_per_pixel = ruler_um / total_ruler_px

    # Confidence based on consistency of spacings
    std_spacing = float(np.std(spacings))
    consistency = 1.0 - min(std_spacing / median_spacing, 1.0) if median_spacing > 0 else 0.0

    return {
        "um_per_pixel": round(um_per_pixel, 4),
        "ruler_px": round(total_ruler_px, 1),
        "tick_spacing_px": round(median_spacing, 1),
        "num_ticks": len(vertical_xs),
        "confidence": round(consistency, 3),
        "method": "auto",
    }


def manual_calibrate(
    image_id: str,
    point1: list[float],
    point2: list[float],
    known_mm: float,
) -> dict:
    """Compute calibration from two user-clicked points.

    Args:
        image_id: uploaded image id
        point1: [x, y] in original image coordinates
        point2: [x, y] in original image coordinates
        known_mm: known real-world distance between the points in mm
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance_px = math.sqrt(dx * dx + dy * dy)

    if distance_px < 1:
        raise ValueError("Points are too close together")

    ruler_um = known_mm * 1000.0
    um_per_pixel = ruler_um / distance_px

    return {
        "um_per_pixel": round(um_per_pixel, 4),
        "ruler_px": round(distance_px, 1),
        "confidence": 1.0,
        "method": "manual",
        "point1": point1,
        "point2": point2,
        "known_mm": known_mm,
    }


def save_calibration(image_id: str, image_stem: str, cal_data: dict) -> str:
    """Persist calibration to JSON. Returns calibration_id."""
    cal_id = uuid.uuid4().hex[:12]
    cal_data["calibration_id"] = cal_id
    cal_data["image_id"] = image_id

    cal_file = settings.calibration_dir / f"{image_stem}_calibration.json"
    settings.calibration_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_file, "w") as f:
        json.dump(cal_data, f, indent=2)

    return cal_id


def get_calibration(image_stem: str) -> dict | None:
    """Load calibration for a given image stem."""
    cal_file = settings.calibration_dir / f"{image_stem}_calibration.json"
    if not cal_file.exists():
        return None
    with open(cal_file) as f:
        return json.load(f)
