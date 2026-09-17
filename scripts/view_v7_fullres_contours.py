#!/usr/bin/env python3
"""Open final full-resolution v7 contour overlays from a pipeline output root."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="View final full-resolution contour overlays")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-stems", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=0.12)
    args = parser.parse_args()

    sources = args.image_stems
    if sources is None or len(sources) == 0:
        sources = sorted(p.name for p in args.output_root.iterdir() if p.is_dir())

    for source in sources:
        path = args.output_root / source / f"{source}_trunk_seg_overlay.jpg"
        image = cv2.imread(str(path))
        if image is None:
            print(f"Missing overlay: {path}")
            continue
        preview = cv2.resize(image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(str(path), preview)
        print(f"Showing {path}. Press any key for next image, Esc to stop.")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:
            break


if __name__ == "__main__":
    main()
