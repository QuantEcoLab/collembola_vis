#!/usr/bin/env python3
"""Detect collembolas in an image and estimate length and volume.

The script segments collembolas using adaptive thresholding + watershed,
then treats each connected component as a specimen. Length is approximated by
its major axis, while volume is estimated by revolving the contour around the
major axis.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from skimage import color, exposure, filters, io, measure, morphology, segmentation, transform
from skimage.feature import peak_local_max
from skimage.morphology import distance_transform_edt


@dataclass
class Measurement:
    label: int
    centroid_x_um: float
    centroid_y_um: float
    length_um: float
    length_mm: float
    volume_um3: float
    volume_mm3: float
    area_um2: float


def segment_collembolas(
    image: np.ndarray,
    *,
    block_size: int,
    offset: float,
    min_area: int,
    disk_radius: int,
) -> np.ndarray:
    """Return labeled mask of collembolas detected in the image."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = color.rgba2rgb(image)
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float64)

    equalized = exposure.equalize_adapthist(gray, clip_limit=0.03)
    local_thresh = filters.threshold_local(equalized, block_size=block_size, offset=offset)
    binary = equalized > local_thresh

    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=min_area)

    if disk_radius > 0:
        selem = morphology.disk(disk_radius)
        binary = morphology.binary_closing(binary, selem)

    distance = distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        labels=binary,
        footprint=np.ones((3, 3), dtype=bool),
        exclude_border=False,
    )

    if coords.size == 0:
        labels = measure.label(binary)
    else:
        markers = np.zeros(distance.shape, dtype=np.int32)
        for idx, (row, col) in enumerate(coords, start=1):
            markers[row, col] = idx
        labels = segmentation.watershed(-distance, markers=markers, mask=binary)

    labels = morphology.remove_small_objects(labels, min_size=min_area)
    return labels.astype(np.int32)


def _principal_axis_angle(mask: np.ndarray) -> float:
    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] < 3:
        return 0.0
    coords = coords.astype(np.float64)
    coords -= coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    axis = vh[0]
    angle = np.degrees(np.arctan2(axis[0], axis[1]))
    return angle


def revolve_volume(mask: np.ndarray, um_per_pixel: float) -> tuple[float, float]:
    angle = _principal_axis_angle(mask)
    rotated = transform.rotate(
        mask.astype(float),
        angle=-angle,
        resize=True,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ) > 0.5

    column_counts = rotated.sum(axis=0)
    nonzero = column_counts > 0
    if not np.any(nonzero):
        return 0.0, 0.0

    radius_um = (column_counts[nonzero] * um_per_pixel) / 2.0
    slice_length_um = um_per_pixel
    volume_um3 = float(np.sum(np.pi * (radius_um ** 2) * slice_length_um))
    axial_length_um = float(nonzero.sum() * um_per_pixel)
    return volume_um3, axial_length_um


def measure_collembolas(labels: np.ndarray, um_per_pixel: float, min_area: int) -> List[Measurement]:
    measurements: List[Measurement] = []
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        component_mask = labels[minr:maxr, minc:maxc] == region.label
        volume_um3, axial_length_um = revolve_volume(component_mask, um_per_pixel)

        length_um = float(region.major_axis_length * um_per_pixel)
        if length_um == 0.0:
            length_um = axial_length_um

        centroid_row, centroid_col = region.centroid
        centroid_x_um = float(centroid_col * um_per_pixel)
        centroid_y_um = float(centroid_row * um_per_pixel)
        area_um2 = float(region.area * (um_per_pixel ** 2))

        measurements.append(
            Measurement(
                label=region.label,
                centroid_x_um=centroid_x_um,
                centroid_y_um=centroid_y_um,
                length_um=length_um,
                length_mm=length_um / 1000.0,
                volume_um3=volume_um3,
                volume_mm3=volume_um3 / 1_000_000_000.0,
                area_um2=area_um2,
            )
        )

    measurements.sort(key=lambda m: m.label)
    return measurements


def write_csv(path: Path, measurements: Iterable[Measurement]) -> None:
    fieldnames = list(Measurement.__annotations__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in measurements:
            writer.writerow(asdict(item))


def print_summary(measurements: List[Measurement]) -> None:
    if not measurements:
        print("No collembolas detected.")
        return

    print(f"Detected {len(measurements)} collembolas\n")
    header = (
        "label",
        "centroid_x_um",
        "centroid_y_um",
        "length_um",
        "volume_um3",
    )
    print("{:>6} {:>14} {:>14} {:>12} {:>14}".format(*header))
    for m in measurements:
        print(
            f"{m.label:6d} "
            f"{m.centroid_x_um:14.1f} "
            f"{m.centroid_y_um:14.1f} "
            f"{m.length_um:12.1f} "
            f"{m.volume_um3:14.1f}"
        )

    total_length = sum(m.length_um for m in measurements)
    total_volume = sum(m.volume_um3 for m in measurements)
    print("\nTotals:")
    print(f"  Length: {total_length:.1f} µm ({total_length / 1000.0:.3f} mm)")
    print(f"  Volume: {total_volume:.1f} µm³ ({total_volume / 1_000_000_000.0:.6f} mm³)")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and measure collembolas.")
    parser.add_argument("image", type=Path, help="Path to the source image (JPEG/PNG).")
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=8.57,
        help="Microns per pixel calibration factor (default: 8.57).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum region area in pixels to keep (default: 50).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=51,
        help="Odd window size for adaptive thresholding (default: 51).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.01,
        help="Offset for adaptive thresholding (default: 0.01).",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=2,
        help="Radius of disk structuring element for binary closing (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write per-collembola measurements.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional JSON file to write summary measurements.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image = io.imread(str(args.image))
    labels = segment_collembolas(
        image,
        block_size=args.block_size,
        offset=args.offset,
        min_area=args.min_area,
        disk_radius=args.closing_radius,
    )

    measurements = measure_collembolas(labels, args.um_per_pixel, min_area=args.min_area)

    print_summary(measurements)

    if args.output:
        write_csv(args.output, measurements)
        print(f"\nWrote CSV to {args.output}")

    if args.json:
        payload = {
            "image": str(args.image),
            "um_per_pixel": args.um_per_pixel,
            "total_length_um": sum(m.length_um for m in measurements),
            "total_volume_um3": sum(m.volume_um3 for m in measurements),
            "measurements": [asdict(m) for m in measurements],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON to {args.json}")


if __name__ == "__main__":
    main()
