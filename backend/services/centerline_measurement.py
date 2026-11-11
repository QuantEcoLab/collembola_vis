"""Application adapter for the approved curved-centerline/circular-section method."""
import math

import cv2
import numpy as np
from skimage.measure import perimeter

from scripts.experimental_centerline import Config, analyze

METHOD = "centerline_circular_sections_v2"
FORMULA = "V = sum(pi * (local_width_i / 2)^2 * delta_length_i)"
WARNING_TEXT = {
    "CROP_EDGE_CONTACT": "Maska dodiruje rub cropa",
    "MULTIPLE_COMPONENTS": "Maska ima odvojene komponente",
    "EMPTY_MASK": "Prazna maska",
    "BODY_EXTRACTION_EMPTY": "Nije izdvojeno stabilno tijelo",
    "EXCESSIVE_BODY_REMOVAL": "Izdvajanje uklonilo velik dio tijela",
    "BODY_EXTRACTION_SPLIT": "Izdvajanje je razdvojilo tijelo",
    "BODY_TOO_THIN_FOR_STABLE_WIDTH": "Tijelo je pretanko za stabilnu širinu",
    "NO_RELIABLE_MAIN_PATH": "Nema pouzdane glavne putanje",
    "SKELETON_LOOP_AMBIGUITY": "Petlja skeleta: nejasna glavna putanja",
    "DISCONNECTED_BODY_SKELETON": "Skelet tijela nije povezan",
    "MAIN_PATH_BRANCH_AMBIGUITY": "Preostale račve glavne putanje",
    "SMOOTHED_PATH_OUTSIDE_BODY": "Središnja linija izlazi iz tijela",
    "BODY_ENDPOINT_UNCERTAIN": "Nepouzdan kraj tijela",
    "LOCAL_WIDTH_SPIKE_POSSIBLE_APPENDAGE": "Skok lokalne širine: mogući izdanak",
    "OFF_CENTER_PATH_OR_APPENDAGE_WIDTH": "Asimetrični presjeci ili bočni izdanak",
    "INVALID_PERPENDICULAR_CROSS_SECTIONS": "Nevaljani okomiti presjeci",
    "BRANCH_TERRITORY_CUT_AMBIGUOUS": "Nejasno odvajanje područja grana",
    "SUBSTANTIAL_BRANCH_TERRITORY_REMOVAL": "Uklonjen velik dio područja grana",
    "ANALYSIS_FAILED": "Mjerenje maske nije uspjelo",
}


def measure_mask(mask, record, calibration):
    """Return current-method row and global-coordinate diagnostic geometry."""
    result = analyze(mask, calibration, Config())
    codes = result["warnings"]
    finite = result["length_px"] is not None and result["volume_px3"] is not None and result["length_px"] > 0 and result["volume_px3"] > 0
    reliable = not result["geometry_unreliable"] and finite
    included = reliable and calibration is not None
    scale = calibration / 1000 if calibration is not None else None
    x, y = record["crop_box"][:2]
    body = result["body_mask"]
    cy, cx = np.nonzero(body if body.any() else mask)
    width = result.get("width_median_px")
    row = {
        "detection_id": record["detection_id"], "segmentation_confidence": record.get("segmentation_confidence"),
        "annotation_id": record.get("annotation_id"),
        "measurement_method": METHOD, "volume_method": METHOD,
        "length_px": result["length_px"], "length_mm": result["length_mm"],
        "width_px": width, "width_mm": width * scale if width is not None and scale is not None else None,
        "width_min_px": result.get("width_min_px"), "width_max_px": result.get("width_max_px"),
        "width_definition": "median local width perpendicular to cleaned body centerline",
        "volume_px3": result["volume_px3"], "volume_mm3": result["volume_mm3"],
        "volume_unit": "mm³" if result["volume_mm3"] is not None else "nije dostupan bez kalibracije" if calibration is None else "nije dostupan: nepouzdana putanja ili presjeci",
        "measurement_reliable": bool(reliable), "volume_included": bool(included),
        "measurement_status": "pouzdano" if reliable else "nepouzdano",
        "warning_codes": "; ".join(codes), "warning": "; ".join(WARNING_TEXT.get(c, c) for c in codes),
        "area_px": int(body.sum()), "area_mm2": float(body.sum() * scale**2) if scale is not None else None,
        "mask_area_px": int(mask.sum()), "perimeter_px": float(perimeter(body)),
        "perimeter_mm": float(perimeter(body) * scale) if scale is not None else None,
        "calibration_um_per_px": calibration, "dedup_status": "kept",
        "touches_crop_edge": "CROP_EDGE_CONTACT" in codes, "mask_components": len(result["component_areas_px"]),
        "centroid_x_px": float(cx.mean() + x), "centroid_y_px": float(cy.mean() + y),
        "model_name": "exp03_no_os", "local_section_count": len(result["cross_sections"]),
    }
    paths, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    def global_point(point):
        return [float(point[0] + x), float(point[1] + y)]
    sections = []
    for s in result["cross_sections"]:
        sections.append({"detection_id": record["detection_id"], "s_px": float(s["s_px"]),
                         "x_px": float(s["x_px"] + x), "y_px": float(s["y_px"] + y),
                         "width_px": s["width_px"], "delta_l_px": s["delta_l_px"], "volume_px3": s["volume_px3"],
                         "width_mm": s["width_px"] * scale if s["width_px"] is not None and scale is not None else None,
                         "delta_l_mm": s["delta_l_px"] * scale if scale is not None else None,
                         "volume_mm3": s["volume_px3"] * scale**3 if s["volume_px3"] is not None and scale is not None else None,
                         "volume_included": bool(included),
                         "left": global_point(s["left_xy"]) if s["left_xy"] is not None else None,
                         "right": global_point(s["right_xy"]) if s["right_xy"] is not None else None})
    geometry = {"detection_id": record["detection_id"], "bbox": record["crop_box"],
                "contours": [[global_point(p) for p in c[:, 0, :]] for c in paths if len(c) >= 3],
                "centerline": [global_point(p) for p in result["centerline_xy"]],
                "measurement_reliable": bool(reliable), "warning": row["warning"],
                "removed_branches": [[global_point(p) for p in b["points_xy"]] for b in result["removed_branches"]],
                "cross_sections": sections}
    return row, geometry


def summarize(rows, calibration):
    eligible = [r for r in rows if r["volume_included"] and r["volume_mm3"] is not None and math.isfinite(r["volume_mm3"])]
    volumes = [r["volume_mm3"] for r in eligible]
    return {"available": calibration is not None, "count": len(volumes), "unit": "mm³" if calibration is not None else None,
            "candidate_count": len(rows), "reliable_count": sum(r["measurement_reliable"] for r in rows),
            "excluded_count": sum(not r["measurement_reliable"] for r in rows),
            "uncalibrated_count": sum(r["measurement_reliable"] and calibration is None for r in rows),
            "mean": float(np.mean(volumes)) if volumes else None,
            "median": float(np.median(volumes)) if volumes else None,
            "min": min(volumes) if volumes else None, "max": max(volumes) if volumes else None,
            "total": float(sum(volumes)) if calibration is not None else None, "measurement_method": METHOD}
