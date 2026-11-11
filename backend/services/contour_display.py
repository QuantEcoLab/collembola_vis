"""Render ordinary external mask outlines in memory; never write result files."""
import json
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings


def draw_outer_contours(result):
    records_path = Path(result["mask_records_path"]).resolve()
    if not records_path.is_relative_to(settings.outputs_dir.resolve()):
        raise ValueError("Invalid mask records path")
    image = cv2.imread(str(result["image_path"]))
    if image is None:
        raise FileNotFoundError("Izvorna fotografija nije dostupna za prikaz kontura.")
    records = json.loads(records_path.read_text())
    for record in records:
        if not record["kept_after_dedup"] or not record["has_mask"]:
            continue
        path = Path(record["mask_path"]).resolve()
        if not path.is_relative_to(settings.outputs_dir.resolve()):
            raise ValueError("Invalid mask path")
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError("Spremljena maska nije dostupna za prikaz kontura.")
        # Display the saved segmentation's external boundary, not the extracted
        # body/skeleton/measurement cross-sections. All disconnected outer paths
        # remain visible, but inner hole boundaries are intentionally omitted.
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset = np.array(record["crop_box"][:2], dtype=np.int32)
        global_contours = [contour + offset for contour in contours]
        cv2.drawContours(image, global_contours, -1, (0, 255, 255), 2)
    return image


def outer_contour_jpeg(result):
    image = draw_outer_contours(result)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Prikaz kontura nije moguće kodirati.")
    return encoded.tobytes()
