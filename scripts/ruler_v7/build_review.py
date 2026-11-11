"""Render saved raw masks only. No model imports, inference, or deduplication."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7"
BASE = OUT.parent / "final_pipeline_v5_vs_v7"
# Full-image pixel coordinates, including the scale's numbers and nearby bodies.
REGIONS = {
    "C_3_2_Collembola011": (4000, 1850, 10200, 4100),
    "C_4_1_PS_500-1000_014": (3350, 1650, 10100, 3900),
    "C_5_1_PS_500-1000_017": (3500, 1650, 10200, 4000),
    "C_1_2_Collembola005": (3850, 1900, 10200, 4050),
    "C_1_3_Collembola006": (3800, 1700, 10200, 4150),
}


def font(size):
    for name in (
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_rows(source):
    path = BASE / source / "detector" / f"{source}_detections_after_bbox_nms.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paths_for(source, det_id, model):
    stem = f"{source}_det{det_id:04d}"
    model_dir = BASE / source / model
    return (model_dir / "all_crops" / f"{stem}.png",
            model_dir / "all_masks" / f"{stem}_mask.png")


def intersects(row, rect):
    return (float(row["x1"]) < rect[2] and float(row["x2"]) > rect[0]
            and float(row["y1"]) < rect[3] and float(row["y2"]) > rect[1])


def inspect():
    report = {}
    # Absent raw mask files are expected only for NO MASK predictions. All raw
    # overlays and crop pairs must exist, including those absent mask cases.
    for source, region in REGIONS.items():
        rows = read_rows(source)
        info = {"bbox_count": len(rows), "region_xyxy": region, "ruler_candidates": 0,
                "v5_raw_masks": 0, "v7_raw_masks": 0, "both_masks": 0,
                "one_mask": 0, "neither_mask": 0}
        for row in rows:
            det_id = int(row["detection_id"])
            c5, m5 = paths_for(source, det_id, "v5_last")
            c7, m7 = paths_for(source, det_id, "v7_best")
            assert c5.is_file() and c7.is_file(), (source, det_id, "missing crop")
            assert c5.read_bytes() == c7.read_bytes(), (source, det_id, "unequal crops")
            for model in ("v5_last", "v7_best"):
                overlay = BASE / source / model / "all_overlays" / f"{source}_det{det_id:04d}_overlay.png"
                assert overlay.is_file(), (source, det_id, "missing raw overlay")
            present = []
            for path in (m5, m7):
                present.append(path.exists())
                if path.exists():
                    with Image.open(path) as mask, Image.open(c5) as crop:
                        assert mask.size == crop.size, (source, det_id, "mask shape")
                        assert mask.getbbox(), (source, det_id, "empty saved mask")
            info["v5_raw_masks"] += int(present[0])
            info["v7_raw_masks"] += int(present[1])
            if intersects(row, region):
                info["ruler_candidates"] += 1
                info["both_masks"] += int(all(present))
                info["one_mask"] += int(sum(present) == 1)
                info["neither_mask"] += int(not any(present))
        report[source] = info
    return report


def region_overview():
    # Reconstruct context from the saved, unmarked full side-by-side left panel.
    # Only for checking ROI placement; comparisons themselves use saved bbox crops.
    overview = Image.new("RGB", (1000, 5 * 400), "white")
    draw = ImageDraw.Draw(overview)
    for idx, (source, region) in enumerate(REGIONS.items()):
        side = BASE / source / "side_by_side" / f"{source}_full_side_by_side.jpg"
        with Image.open(side) as img:
            panel_w = img.width // 3
            original = img.crop((0, 38, panel_w, img.height)).convert("RGB")
        scale = min(420 / original.width, 340 / original.height)
        original = original.resize((int(original.width * scale), int(original.height * scale)))
        y = idx * 400 + 28
        overview.paste(original, (0, y))
        draw.text((8, idx * 400 + 4), source, font=font(18), fill="black")
        x1, y1, x2, y2 = region
        fx = original.width / 10497
        fy = original.height / 10880
        draw.rectangle((x1 * fx, y + y1 * fy, x2 * fx, y + y2 * fy), outline="red", width=2)
        region_img = original.crop((int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy)))
        region_img = region_img.resize((530, max(1, round(region_img.height * 530 / region_img.width))))
        overview.paste(region_img, (450, y + 20))
    overview.save(OUT / "ruler_region_overview.jpg", quality=94)


def render(crop_path, mask_paths, target):
    with Image.open(crop_path) as image:
        crop = np.array(image.convert("RGB"))
    h, w = crop.shape[:2]
    scale = min(390 / w, 340 / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    original = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    panels = [original.copy(), original.copy(), original.copy()]
    masks = []
    for idx, path in enumerate(mask_paths, start=1):
        if not path.exists():
            masks.append(None)
            continue
        with Image.open(path) as image:
            mask = np.array(image.convert("L")) > 0
        masks.append(mask)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [np.round(contour.astype(float) * scale).astype(np.int32) for contour in contours]
        cv2.drawContours(panels[idx], contours, -1, (255, 215, 0), 2, cv2.LINE_AA)
    canvas = Image.new("RGB", (1200, 390), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, title) in enumerate(zip(panels, ("original", "v5", "v7"))):
        draw.text((idx * 400 + 12, 6), title, fill="black", font=font(24))
        canvas.paste(Image.fromarray(panel), (idx * 400 + (400 - size[0]) // 2, 42 + (340 - size[1]) // 2))
    canvas.save(target)
    if all(mask is not None for mask in masks):
        union = np.logical_or(*masks).sum()
        iou = float(np.logical_and(*masks).sum() / union) if union else 1.0
        area_ratio = float(min(m.sum() for m in masks) / max(m.sum() for m in masks))
    else:
        iou, area_ratio = None, None
    return iou, area_ratio


def sheets(items, prefix):
    paths = []
    per_page = 12
    for start in range(0, len(items), per_page):
        page_items = items[start:start + per_page]
        canvas = Image.new("RGB", (2400, math.ceil(len(page_items) / 2) * 430 + 48), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{prefix} / {start // per_page + 1}", fill="black", font=font(24))
        for idx, item in enumerate(page_items):
            x, y = (idx % 2) * 1200, 48 + (idx // 2) * 430
            draw.text((x + 12, y + 2), f"{item['source_id']} / {item['detection_id']}", fill="black", font=font(20))
            with Image.open(ROOT / item["comparison_path"]) as image:
                canvas.paste(image, (x, y + 30))
        path = OUT / f"{prefix}_{start // per_page + 1:03d}.jpg"
        canvas.save(path, quality=95)
        paths.append(relative(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    report = inspect()
    if args.inspect:
        print(json.dumps(report, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    region_overview()
    (OUT / "comparisons").mkdir(exist_ok=True)
    index = []
    for source, region in REGIONS.items():
        for row in read_rows(source):
            if not intersects(row, region):
                continue
            det_id = int(row["detection_id"])
            crop, m5 = paths_for(source, det_id, "v5_last")
            _, m7 = paths_for(source, det_id, "v7_best")
            target = OUT / "comparisons" / f"{source}_det{det_id:04d}.png"
            iou, ratio = render(crop, (m5, m7), target)
            one_mask = m5.exists() != m7.exists()
            different = one_mask or (iou is not None and (iou < 0.80 or ratio < 0.75))
            index.append({"source_id": source, "detection_id": det_id,
                          "original_path": relative(crop),
                          "v5_mask_path": relative(m5) if m5.exists() else "",
                          "v7_mask_path": relative(m7) if m7.exists() else "",
                          "comparison_path": relative(target),
                          "v5_has_mask": m5.exists(), "v7_has_mask": m7.exists(),
                          "difference_review_candidate": different,
                          "mask_pixel_iou_for_sorting": iou,
                          "bbox_x1": row["x1"], "bbox_y1": row["y1"],
                          "bbox_x2": row["x2"], "bbox_y2": row["y2"]})
    differences = sorted([item for item in index if item["difference_review_candidate"]],
                         key=lambda item: (-int(item["v5_has_mask"] != item["v7_has_mask"]),
                                           item["mask_pixel_iou_for_sorting"] if item["mask_pixel_iou_for_sorting"] is not None else -1))
    diff_pages = sheets(differences, "01_differences")
    all_pages = sheets(index, "02_all_ruler")
    for name, rows in (("index.csv", index), ("differences_index.csv", differences)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"raw_inventory": report, "ruler_detection_count": len(index),
               "both_masks": sum(item["v5_has_mask"] and item["v7_has_mask"] for item in index),
               "one_mask": sum(item["v5_has_mask"] != item["v7_has_mask"] for item in index),
               "neither_mask": sum(not item["v5_has_mask"] and not item["v7_has_mask"] for item in index),
               "difference_candidates": len(differences),
               "contact_sheets_in_review_order": diff_pages + all_pages,
               "selection": "Any saved bbox intersecting the documented ruler-and-nearby-body rectangle. No inference or dedup. Difference flags are review candidates, not anatomical correctness judgments.",
               "difference_rule": "one model lacks mask, or mask IoU < 0.80, or area ratio < 0.75"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Raw Contour Review", "", "Source: saved all_crops and all_masks before dedup. No models run.",
             "All bbox IDs are included regardless of later retention. Blank mask paths mean no saved raw mask.",
             "Same crop, scale, contour colour and thickness in every comparison. No metrics on image panels.",
             "Difference sheets prioritize missing-mask and large mask disagreement cases, not a winning model.",
             "ROI selection includes ticks, digits and nearby bodies; see ruler_region_overview.jpg and summary.json.",
             "", "## Review Order"]
    lines.extend(f"{idx}. `{path}`" for idx, path in enumerate(diff_pages + all_pages, 1))
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
