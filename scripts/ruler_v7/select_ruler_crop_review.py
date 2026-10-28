#!/usr/bin/env python3
"""Review-only rectangle selector for ruler LabelMe crops."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def load_labelme(path: Path) -> tuple[dict[str, Any], Path, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_path = Path(data["imagePath"].replace("\\", "/"))
    if not image_path.is_absolute():
        image_path = (path.parent / image_path).resolve()
    Image.MAX_IMAGE_PIXELS = None
    image = cv2.cvtColor(np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return data, image_path, image


def pts(shape: dict[str, Any]) -> np.ndarray:
    return np.asarray(shape.get("points", []), dtype=np.float32)


def bbox(shape: dict[str, Any]) -> tuple[float, float, float, float]:
    p = pts(shape)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def inside(shape: dict[str, Any], rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    p = pts(shape)
    return len(p) >= 3 and bool(np.all(p[:, 0] >= x1) and np.all(p[:, 0] <= x2) and np.all(p[:, 1] >= y1) and np.all(p[:, 1] <= y2))


def intersects(shape: dict[str, Any], rect: tuple[int, int, int, int]) -> bool:
    bx1, by1, bx2, by2 = bbox(shape)
    x1, y1, x2, y2 = rect
    return bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1


def split_shapes(shapes: list[dict[str, Any]], rect: tuple[int, int, int, int]) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    kept, cut = [], []
    for i, shape in enumerate(shapes, 1):
        if inside(shape, rect):
            kept.append((i, shape))
        elif intersects(shape, rect):
            cut.append((i, shape))
    return kept, cut


def draw_shapes(img: np.ndarray, shapes: list[dict[str, Any]], color: tuple[int, int, int], offset: tuple[int, int] = (0, 0), labels: bool = False) -> None:
    ox, oy = offset
    overlay = img.copy()
    for i, shape in enumerate(shapes, 1):
        p = pts(shape)
        if len(p) < 3:
            continue
        q = np.round(p - np.array([[ox, oy]], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(overlay, [q], color)
        cv2.polylines(img, [q], True, color, 2, cv2.LINE_AA)
        if labels:
            c = q.mean(axis=0).astype(int)
            draw_text(img, str(i), (int(c[0]), int(c[1])), 0.45)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def shifted(shape: dict[str, Any], rect: tuple[int, int, int, int], source_index: int) -> dict[str, Any]:
    x1, y1, _x2, _y2 = rect
    out = dict(shape)
    out["points"] = [[float(x) - x1, float(y) - y1] for x, y in shape.get("points", [])]
    out["flags"] = dict(shape.get("flags") or {})
    out["flags"]["source_polygon_index_1based"] = source_index
    return out


def fit(image: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


class Selector:
    def __init__(self, image: np.ndarray, shapes: list[dict[str, Any]], max_w: int, max_h: int):
        self.image = image
        self.shapes = shapes
        self.max_w = max_w
        self.max_h = max_h
        self.rect: tuple[int, int, int, int] | None = None
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.dragging = False
        _display, self.scale = fit(image, max_w, max_h)

    def to_image(self, x: int, y: int) -> tuple[int, int]:
        ix = min(max(int(round(x / self.scale)), 0), self.image.shape[1] - 1)
        iy = min(max(int(round(y / self.scale)), 0), self.image.shape[0] - 1)
        return ix, iy

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.to_image(x, y)
            self.current = self.start
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = self.to_image(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            end = self.to_image(x, y)
            x1, x2 = sorted([self.start[0], end[0]])
            y1, y2 = sorted([self.start[1], end[1]])
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                self.rect = (x1, y1, x2, y2)
            self.dragging = False

    def render(self) -> np.ndarray:
        canvas = self.image.copy()
        draw_shapes(canvas, self.shapes, YELLOW, labels=True)
        rect = self.rect
        if self.dragging and self.start and self.current:
            x1, x2 = sorted([self.start[0], self.current[0]])
            y1, y2 = sorted([self.start[1], self.current[1]])
            rect = (x1, y1, x2, y2)
        if rect:
            kept, cut = split_shapes(self.shapes, rect)
            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), CYAN, 3, cv2.LINE_AA)
            draw_text(canvas, f"rect={rect} kept={len(kept)} cut_labeled={len(cut)}", (16, 34))
        draw_text(canvas, "Drag rectangle. ENTER accept. C clear. Q/Esc exit.", (16, canvas.shape[0] - 24))
        return fit(canvas, self.max_w, self.max_h)[0]


class MarkerReview:
    def __init__(self, crop: np.ndarray, kept_shapes: list[dict[str, Any]], cut_shapes: list[dict[str, Any]], rect: tuple[int, int, int, int], max_w: int, max_h: int):
        self.crop = crop
        self.kept_shapes = kept_shapes
        self.cut_shapes = cut_shapes
        self.rect = rect
        self.max_w = max_w
        self.max_h = max_h
        _display, self.scale = fit(crop, max_w, max_h)
        self.markers: list[tuple[int, int]] = []

    def to_crop(self, x: int, y: int) -> tuple[int, int]:
        ix = min(max(int(round(x / self.scale)), 0), self.crop.shape[1] - 1)
        iy = min(max(int(round(y / self.scale)), 0), self.crop.shape[0] - 1)
        return ix, iy

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.markers.append(self.to_crop(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.markers:
            self.markers.pop()

    def overlay(self) -> np.ndarray:
        canvas = self.crop.copy()
        draw_shapes(canvas, self.kept_shapes, YELLOW)
        draw_shapes(canvas, self.cut_shapes, RED, offset=(self.rect[0], self.rect[1]))
        for x, y in self.markers:
            cv2.drawMarker(canvas, (x, y), RED, cv2.MARKER_TILTED_CROSS, 30, 3, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 16, RED, 2, cv2.LINE_AA)
        draw_text(canvas, "Click cut UNLABELED edge organisms. Right-click undo. S save. R redo. Q exit.", (16, 32), 0.55)
        return canvas

    def render(self) -> np.ndarray:
        return fit(self.overlay(), self.max_w, self.max_h)[0]


def edge_strips(crop: np.ndarray, strip: int = 140) -> np.ndarray:
    h, w = crop.shape[:2]
    panels = [
        ("TOP", crop[: min(strip, h), :]),
        ("BOTTOM", crop[max(0, h - strip):h, :]),
        ("LEFT rotated", cv2.rotate(crop[:, : min(strip, w)], cv2.ROTATE_90_CLOCKWISE)),
        ("RIGHT rotated", cv2.rotate(crop[:, max(0, w - strip):w], cv2.ROTATE_90_CLOCKWISE)),
    ]
    rows = []
    for label, panel in panels:
        p, _s = fit(panel, 2200, 160)
        label_img = np.full((p.shape[0], 180, 3), 255, np.uint8)
        cv2.putText(label_img, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)
        rows.append(np.hstack([label_img, p]))
    width = max(r.shape[1] for r in rows)
    return np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])


def save(out_root: Path, source_json: Path, source_image: Path, payload: dict[str, Any], image: np.ndarray, rect: tuple[int, int, int, int], kept: list[tuple[int, dict[str, Any]]], cut: list[tuple[int, dict[str, Any]]], markers: list[tuple[int, int]]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / f"selection_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    x1, y1, x2, y2 = rect
    crop = image[y1:y2, x1:x2].copy()
    crop_name = f"crop_x{x1}_y{y1}_w{x2-x1}_h{y2-y1}.png"
    cv2.imwrite(str(out / crop_name), crop)
    kept_shapes = [shifted(shape, rect, idx) for idx, shape in kept]
    cut_shapes = [shape for _idx, shape in cut]
    review = MarkerReview(crop, kept_shapes, cut_shapes, rect, 99999, 99999)
    review.markers = markers
    cv2.imwrite(str(out / "overlay_review.jpg"), review.overlay(), [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(out / "edge_strips_review.jpg"), edge_strips(crop), [cv2.IMWRITE_JPEG_QUALITY, 94])
    selected = {k: v for k, v in payload.items() if k not in {"imageData", "imagePath", "imageWidth", "imageHeight", "shapes"}}
    selected.update({"imagePath": crop_name, "imageData": None, "imageWidth": x2 - x1, "imageHeight": y2 - y1, "shapes": kept_shapes})
    selected["flags"] = dict(payload.get("flags") or {})
    selected["flags"].update({"source_labelme_json": str(source_json), "source_image": str(source_image), "crop_box_xyxy_in_source_crop": [x1, y1, x2, y2]})
    (out / "selected_labelme.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    meta = {
        "source_labelme_json": str(source_json),
        "source_image": str(source_image),
        "crop_box_xyxy_in_source_crop": [x1, y1, x2, y2],
        "crop_size": [x2 - x1, y2 - y1],
        "kept_polygon_count": len(kept),
        "kept_source_polygon_indices_1based": [idx for idx, _shape in kept],
        "cut_labeled_polygon_count": len(cut),
        "cut_labeled_source_polygon_indices_1based": [idx for idx, _shape in cut],
        "manual_unlabeled_edge_markers_crop_xy": markers,
        "crop_image": str(out / crop_name),
        "selected_labelme_json": str(out / "selected_labelme.json"),
        "overlay_review": str(out / "overlay_review.jpg"),
        "edge_strips_review": str(out / "edge_strips_review.jpg"),
    }
    (out / "selection_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def run(args: argparse.Namespace) -> None:
    payload, image_path, image = load_labelme(args.labelme_json)
    shapes = payload.get("shapes", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        selector = Selector(image, shapes, args.max_width, args.max_height)
        cv2.namedWindow("select crop rectangle", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("select crop rectangle", selector.mouse)
        while True:
            cv2.imshow("select crop rectangle", selector.render())
            key = cv2.waitKey(30)
            if key in {27, ord("q"), ord("Q")}:
                cv2.destroyAllWindows()
                return
            if key in {ord("c"), ord("C")}:
                selector.rect = None
            if key in {13, 10} and selector.rect is not None:
                break
        rect = selector.rect
        assert rect is not None
        kept, cut = split_shapes(shapes, rect)
        x1, y1, x2, y2 = rect
        crop = image[y1:y2, x1:x2].copy()
        review = MarkerReview(crop, [shifted(shape, rect, idx) for idx, shape in kept], [shape for _idx, shape in cut], rect, args.max_width, args.max_height)
        cv2.namedWindow("review selected crop", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("review selected crop", review.mouse)
        while True:
            cv2.imshow("review selected crop", review.render())
            key = cv2.waitKey(30)
            if key in {27, ord("q"), ord("Q")}:
                cv2.destroyAllWindows()
                return
            if key in {ord("r"), ord("R")}:
                cv2.destroyWindow("review selected crop")
                break
            if key in {ord("s"), ord("S")}:
                out = save(args.output_dir, args.labelme_json, image_path, payload, image, rect, kept, cut, review.markers)
                print(f"Saved selection: {out}")
                print(f"Kept polygons: {len(kept)}; cut labeled polygons: {len(cut)}; manual unlabeled edge markers: {len(review.markers)}")
                cv2.destroyWindow("review selected crop")
                if args.exit_after_save:
                    cv2.destroyAllWindows()
                    return
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively select review-only ruler crop rectangles")
    parser.add_argument("--labelme-json", type=Path, default=Path("data/ruler_manual_annotation_v1/labelme/C_2_1_Collembola007_ruler_0_25__x3500_y1400_w3400_h2500.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ruler_manual_annotation_v1/review/manual_rect_selection"))
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=850)
    parser.add_argument("--exit-after-save", action="store_true", help="Exit after saving one selection instead of returning to rectangle selection.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
