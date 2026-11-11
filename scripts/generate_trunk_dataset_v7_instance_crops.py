from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data/trunk_dataset_combined_v5/yolo_seg"
V7 = ROOT / "data/trunk_dataset_combined_v7/yolo_seg"
RULER = ROOT / "data/ruler_manual_annotation_v1/yolo_seg_review_export_v1/yolo_seg"
STAGING = ROOT / "data/trunk_dataset_combined_v7/instance_crop_generation"

TARGET_OCCUPANCY = 0.75
MIN_OCCUPANCY = 0.65
MAX_OCCUPANCY = 0.85
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Poly:
    source_image: str
    source_stem: str
    polygon_id: int
    points: list[tuple[float, float]]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def parse_label(label_path: Path, image_path: Path) -> list[Poly]:
    with Image.open(image_path) as im:
        w, h = im.size
    polys: list[Poly] = []
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if not parts:
            continue
        coords = [float(v) for v in parts[1:]]
        points = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
        polys.append(Poly(image_path.name, image_path.stem, idx, points))
    return polys


def rect_contains_bbox(rect: tuple[float, float, float, float], bbox: tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    x1, y1, x2, y2 = rect
    bx1, by1, bx2, by2 = bbox
    return bx1 >= x1 - eps and by1 >= y1 - eps and bx2 <= x2 + eps and by2 <= y2 + eps


def rect_intersects_bbox(rect: tuple[float, float, float, float], bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = rect
    bx1, by1, bx2, by2 = bbox
    return x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1


def clamp_rect(cx: float, cy: float, cw: float, ch: float, iw: int, ih: int) -> tuple[float, float, float, float]:
    cw = min(cw, float(iw))
    ch = min(ch, float(ih))
    x1 = cx - cw / 2
    y1 = cy - ch / 2
    x1 = max(0.0, min(x1, iw - cw))
    y1 = max(0.0, min(y1, ih - ch))
    return x1, y1, x1 + cw, y1 + ch


def make_candidate(primary: Poly, all_polys: list[Poly], iw: int, ih: int) -> tuple[str, tuple[float, float, float, float] | None, list[int], str]:
    bx1, by1, bx2, by2 = primary.bbox
    bw = bx2 - bx1
    bh = by2 - by1
    if bw <= 1 or bh <= 1:
        return "rejected", None, [], "primary_bbox_too_small"

    min_w, max_w = bw / MAX_OCCUPANCY, bw / MIN_OCCUPANCY
    min_h, max_h = bh / MAX_OCCUPANCY, bh / MIN_OCCUPANCY
    cw, ch = bw / TARGET_OCCUPANCY, bh / TARGET_OCCUPANCY
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    rect = clamp_rect(cx, cy, cw, ch, iw, ih)
    adjusted = False

    for _ in range(20):
        changed = False
        for other in all_polys:
            obox = other.bbox
            if not rect_intersects_bbox(rect, obox) or rect_contains_bbox(rect, obox):
                continue

            ux1 = min(rect[0], obox[0])
            uy1 = min(rect[1], obox[1])
            ux2 = max(rect[2], obox[2])
            uy2 = max(rect[3], obox[3])
            required_w = max(rect[2] - rect[0], ux2 - ux1)
            required_h = max(rect[3] - rect[1], uy2 - uy1)
            if required_w > max_w or required_h > max_h:
                return "rejected", None, [], "intersecting_polygon_requires_too_large_crop"

            new_rect = clamp_rect((ux1 + ux2) / 2, (uy1 + uy2) / 2, required_w, required_h, iw, ih)
            if not rect_contains_bbox(new_rect, obox) or not rect_contains_bbox(new_rect, primary.bbox):
                return "rejected", None, [], "cannot_shift_crop_to_contain_intersecting_polygon"
            rect = new_rect
            adjusted = True
            changed = True
        if not changed:
            break

    pw = bw / (rect[2] - rect[0])
    ph = bh / (rect[3] - rect[1])
    if not (MIN_OCCUPANCY <= pw <= MAX_OCCUPANCY and MIN_OCCUPANCY <= ph <= MAX_OCCUPANCY):
        return "rejected", None, [], "primary_occupancy_outside_65_85_percent"

    included: list[int] = []
    for other in all_polys:
        obox = other.bbox
        if rect_contains_bbox(rect, obox):
            included.append(other.polygon_id)
        elif rect_intersects_bbox(rect, obox):
            return "rejected", None, [], "final_crop_would_cut_polygon"

    return ("adjusted" if adjusted else "accepted"), rect, included, "ok"


def transform_points(points: list[tuple[float, float]], rect: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = rect
    cw = x2 - x1
    ch = y2 - y1
    return [((x - x1) / cw, (y - y1) / ch) for x, y in points]


def polygon_to_yolo(points: list[tuple[float, float]]) -> str:
    vals: list[str] = ["0"]
    for x, y in points:
        vals.append(f"{min(1.0, max(0.0, x)):.6f}")
        vals.append(f"{min(1.0, max(0.0, y)):.6f}")
    return " ".join(vals)


def draw_audit(src: Image.Image, rect: tuple[float, float, float, float], polys: list[Poly], included: set[int], primary_id: int, title: str, out_path: Path) -> None:
    crop = src.crop(tuple(round(v) for v in rect)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    x1, y1, _, _ = rect
    for p in polys:
        shifted = [(x - x1, y - y1) for x, y in p.points]
        color = "lime" if p.polygon_id == primary_id else ("cyan" if p.polygon_id in included else "red")
        if p.polygon_id in included:
            draw.line(shifted + [shifted[0]], fill=color, width=2)
    draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline="yellow", width=3)
    banner_h = 24
    canvas = Image.new("RGB", (crop.width, crop.height + banner_h), "white")
    canvas.paste(crop, (0, banner_h))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def quantiles(vals: list[float]) -> dict[str, float]:
    vals = sorted(vals)
    if not vals:
        return {}
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac
    return {"min": vals[0], "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "max": vals[-1]}


def label_boxes(label_path: Path, image_path: Path, imgsz: int = 320) -> list[dict[str, float]]:
    with Image.open(image_path) as im:
        iw, ih = im.size
    scale = min(imgsz / iw, imgsz / ih)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(x) for x in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        nw = max(xs) - min(xs)
        nh = max(ys) - min(ys)
        boxes.append({
            "norm_w": nw,
            "norm_h": nh,
            "norm_area": nw * nh,
            "resized_w_px": nw * iw * scale,
            "resized_h_px": nh * ih * scale,
        })
    return boxes


def summarize_boxes(name: str, image_paths: list[Path], label_dir: Path) -> dict[str, object]:
    boxes: list[dict[str, float]] = []
    poly_counts: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as im:
            widths.append(im.width)
            heights.append(im.height)
        current = label_boxes(label_path, image_path)
        boxes.extend(current)
        poly_counts.append(len(current))
    return {
        "group": name,
        "images": len(image_paths),
        "polygons": len(boxes),
        "image_width": quantiles([float(x) for x in widths]),
        "image_height": quantiles([float(x) for x in heights]),
        "polygons_per_image": quantiles([float(x) for x in poly_counts]),
        "norm_bbox_width": quantiles([x["norm_w"] for x in boxes]),
        "norm_bbox_height": quantiles([x["norm_h"] for x in boxes]),
        "norm_bbox_area": quantiles([x["norm_area"] for x in boxes]),
        "resized_bbox_width_px_at_320_letterbox": quantiles([x["resized_w_px"] for x in boxes]),
        "resized_bbox_height_px_at_320_letterbox": quantiles([x["resized_h_px"] for x in boxes]),
    }


def make_contact_sheets(audit_paths: list[Path], out_dir: Path, thumb_w: int = 220, cols: int = 5) -> list[str]:
    sheets: list[str] = []
    for sheet_idx in range(math.ceil(len(audit_paths) / (cols * 5))):
        subset = audit_paths[sheet_idx * cols * 5 : (sheet_idx + 1) * cols * 5]
        thumbs = []
        for path in subset:
            im = Image.open(path).convert("RGB")
            scale = thumb_w / im.width
            thumb = im.resize((thumb_w, max(1, round(im.height * scale))))
            thumbs.append(thumb)
        row_heights = [max(t.height for t in thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        canvas = Image.new("RGB", (thumb_w * cols, sum(row_heights)), "white")
        y = 0
        for row_idx, row_start in enumerate(range(0, len(thumbs), cols)):
            for col_idx, thumb in enumerate(thumbs[row_start:row_start + cols]):
                canvas.paste(thumb, (col_idx * thumb_w, y))
            y += row_heights[row_idx]
        out_path = out_dir / f"contact_sheet_{sheet_idx + 1:03d}.jpg"
        canvas.save(out_path, quality=92)
        sheets.append(str(out_path.relative_to(ROOT)))
    return sheets


def main() -> None:
    if not V5.exists():
        raise SystemExit(f"Missing source dataset: {V5}")
    if V7.parent.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {V7.parent}")

    crops_dir = STAGING / "accepted_crops"
    audit_dir = STAGING / "audit_views"
    crops_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    audit_paths: list[Path] = []
    accepted_images: list[Path] = []

    for image_path in sorted((RULER / "images/train").iterdir()):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = RULER / "labels/train" / f"{image_path.stem}.txt"
        polys = parse_label(label_path, image_path)
        with Image.open(image_path) as src:
            iw, ih = src.size
            for primary in polys:
                status, rect, included_ids, reason = make_candidate(primary, polys, iw, ih)
                candidate_id = f"{image_path.stem}__inst_{primary.polygon_id:03d}"
                if rect is None:
                    rejected_rows.append({
                        "candidate_id": candidate_id,
                        "source_image": image_path.name,
                        "primary_polygon_id": primary.polygon_id,
                        "status": status,
                        "reason": reason,
                    })
                    continue

                crop_path = crops_dir / f"{candidate_id}.png"
                label_out = crops_dir / f"{candidate_id}.txt"
                rounded = tuple(round(v) for v in rect)
                rounded_rect = tuple(float(v) for v in rounded)
                src.crop(rounded).save(crop_path)
                included = [p for p in polys if p.polygon_id in set(included_ids)]
                label_out.write_text("\n".join(polygon_to_yolo(transform_points(p.points, rounded_rect)) for p in included) + "\n", encoding="utf-8")

                bx1, by1, bx2, by2 = primary.bbox
                occupancy_w = (bx2 - bx1) / (rounded_rect[2] - rounded_rect[0])
                occupancy_h = (by2 - by1) / (rounded_rect[3] - rounded_rect[1])
                audit_path = audit_dir / f"{candidate_id}.jpg"
                draw_audit(src, rounded_rect, polys, set(included_ids), primary.polygon_id, f"{candidate_id} {status} occ={occupancy_w:.2f}x{occupancy_h:.2f}", audit_path)
                audit_paths.append(audit_path)
                accepted_images.append(crop_path)
                manifest_rows.append({
                    "candidate_id": candidate_id,
                    "source_image": image_path.name,
                    "source_label": label_path.name,
                    "primary_polygon_id": primary.polygon_id,
                    "status": status,
                    "reason": reason,
                    "included_polygon_ids": ";".join(str(x) for x in included_ids),
                    "included_polygon_count": len(included_ids),
                    "source_image_width": iw,
                    "source_image_height": ih,
                    "crop_x1": rounded[0],
                    "crop_y1": rounded[1],
                    "crop_x2": rounded[2],
                    "crop_y2": rounded[3],
                    "crop_width": rounded[2] - rounded[0],
                    "crop_height": rounded[3] - rounded[1],
                    "primary_bbox_occupancy_w": round(occupancy_w, 6),
                    "primary_bbox_occupancy_h": round(occupancy_h, 6),
                    "target_image": f"images/train/{crop_path.name}",
                    "target_label": f"labels/train/{label_out.name}",
                    "audit_view": str(audit_path.relative_to(ROOT)),
                })

    contact_sheets = make_contact_sheets(audit_paths, STAGING)

    shutil.copytree(V5, V7)
    (V7 / "dataset.yaml").write_text(
        f"path: {V7.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: trunk\n",
        encoding="utf-8",
    )

    existing_names = {p.name for p in (V7 / "images/train").iterdir()} | {p.name for p in (V7 / "labels/train").iterdir()}
    collisions = []
    for row in manifest_rows:
        image_name = Path(str(row["target_image"])).name
        label_name = Path(str(row["target_label"])).name
        if image_name in existing_names or label_name in existing_names:
            collisions.append({"image": image_name, "label": label_name})
            continue
        shutil.copy2(crops_dir / image_name, V7 / "images/train" / image_name)
        shutil.copy2(crops_dir / label_name, V7 / "labels/train" / label_name)

    with (V7 / "added_instance_crop_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (V7 / "rejected_instance_crop_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["candidate_id", "source_image", "primary_polygon_id", "status", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected_rows)

    v5_train_images = [p for p in sorted((V5 / "images/train").iterdir()) if p.suffix.lower() in IMG_EXTS]
    v7_added_images = [V7 / str(row["target_image"]) for row in manifest_rows]
    scale_audit = {
        "imgsz": 320,
        "resize_assumption": "Ultralytics letterbox scale: scale=min(imgsz/image_width, imgsz/image_height); padding ignored for object size.",
        "groups": [
            summarize_boxes("v5_train_original", v5_train_images, V5 / "labels/train"),
            summarize_boxes("v7_added_instance_centered_ruler_crops", v7_added_images, V7 / "labels/train"),
        ],
    }
    (V7 / "scale_audit_summary.json").write_text(json.dumps(scale_audit, indent=2), encoding="utf-8")

    def count_files(path: Path, suffixes: set[str] | None = None) -> int:
        files = [p for p in path.iterdir() if p.is_file()]
        if suffixes:
            files = [p for p in files if p.suffix.lower() in suffixes]
        return len(files)

    invalid_labels = []
    missing_pairs = {"train_images_without_labels": [], "train_labels_without_images": [], "val_images_without_labels": [], "val_labels_without_images": []}
    for split in ["train", "val"]:
        images = {p.stem for p in (V7 / "images" / split).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        labels = {p.stem for p in (V7 / "labels" / split).iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        missing_pairs[f"{split}_images_without_labels"] = sorted(images - labels)
        missing_pairs[f"{split}_labels_without_images"] = sorted(labels - images)
        for label_path in (V7 / "labels" / split).iterdir():
            if label_path.suffix.lower() != ".txt":
                continue
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_labels.append({"file": str(label_path.relative_to(ROOT)), "line": line_no, "reason": "parse_error"})
                    continue
                if cls != 0 or len(coords) < 6 or len(coords) % 2 != 0 or any(x < 0 or x > 1 for x in coords):
                    invalid_labels.append({"file": str(label_path.relative_to(ROOT)), "line": line_no, "reason": "invalid_class_or_coordinates"})

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(V5.relative_to(ROOT)),
        "target_dataset": str(V7.relative_to(ROOT)),
        "ruler_source_dataset": str(RULER.relative_to(ROOT)),
        "large_ruler_scenes_added_to_train": False,
        "candidate_counts": {
            "total_primary_polygons": len(manifest_rows) + len(rejected_rows),
            "accepted_total": len(manifest_rows),
            "accepted_without_adjustment": sum(1 for r in manifest_rows if r["status"] == "accepted"),
            "adjusted": sum(1 for r in manifest_rows if r["status"] == "adjusted"),
            "rejected": len(rejected_rows),
        },
        "dataset_counts": {
            "train_images": count_files(V7 / "images/train", IMG_EXTS),
            "train_labels": count_files(V7 / "labels/train", {".txt"}),
            "val_images": count_files(V7 / "images/val", IMG_EXTS),
            "val_labels": count_files(V7 / "labels/val", {".txt"}),
        },
        "checks": {
            "no_name_collisions": len(collisions) == 0,
            "no_missing_pairs": all(len(v) == 0 for v in missing_pairs.values()),
            "no_invalid_labels": len(invalid_labels) == 0,
            "v5_val_preserved_count": count_files(V7 / "images/val", IMG_EXTS) == count_files(V5 / "images/val", IMG_EXTS) and count_files(V7 / "labels/val", {".txt"}) == count_files(V5 / "labels/val", {".txt"}),
        },
        "collisions": collisions,
        "missing_pairs": missing_pairs,
        "invalid_labels": invalid_labels,
        "audit_views_dir": str(audit_dir.relative_to(ROOT)),
        "contact_sheets": contact_sheets,
        "manifest": str((V7 / "added_instance_crop_manifest.csv").relative_to(ROOT)),
        "rejected_candidates": str((V7 / "rejected_instance_crop_candidates.csv").relative_to(ROOT)),
        "scale_audit": str((V7 / "scale_audit_summary.json").relative_to(ROOT)),
    }
    (V7 / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
