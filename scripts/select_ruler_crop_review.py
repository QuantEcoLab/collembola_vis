#!/usr/bin/env python3
"""Interactively select rectangular review crops from an image.

Review-only tool: writes only the selected PNG crop and metadata JSON into a
timestamped output directory. It does not create LabelMe files, modify datasets,
or start training.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    return min(1.0, max_width / max(1, width), max_height / max(1, height))


def save_image_selection(
    image: Image.Image,
    image_path: Path,
    rect: tuple[int, int, int, int],
    output_root: Path,
) -> Path:
    x1, y1, x2, y2 = rect
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"selection_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    crop = image.crop((x1, y1, x2, y2))
    crop_name = f"{image_path.stem}__x{x1}_y{y1}_w{x2 - x1}_h{y2 - y1}.png"
    crop_path = out_dir / crop_name
    crop.save(crop_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_image": str(image_path),
        "source_width": int(image.width),
        "source_height": int(image.height),
        "crop_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "crop_x": int(x1),
        "crop_y": int(y1),
        "crop_width": int(x2 - x1),
        "crop_height": int(y2 - y1),
        "crop_image": str(crop_path),
        "note": "Review-only crop. No LabelMe JSON, dataset, or training output was created.",
    }
    (out_dir / "selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


class CropSelector:
    def __init__(self, image_path: Path, output_dir: Path, max_width: int, max_height: int) -> None:
        Image.MAX_IMAGE_PIXELS = None
        self.image_path = image_path.resolve()
        self.output_dir = output_dir
        self.image = Image.open(self.image_path).convert("RGB")
        self.scale = fit_scale(self.image.width, self.image.height, max_width, max_height)
        display_size = (max(1, round(self.image.width * self.scale)), max(1, round(self.image.height * self.scale)))
        self.display_image = self.image.resize(display_size, Image.Resampling.LANCZOS)

        self.root = tk.Tk()
        self.root.title("Select crop rectangle")
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas = tk.Canvas(self.root, width=display_size[0], height=display_size[1], cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.status = tk.StringVar(value="Drag rectangle. Enter saves. C clears. Q/Esc exits.")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")

        self.start: tuple[int, int] | None = None
        self.rect_canvas_id: int | None = None
        self.rect: tuple[int, int, int, int] | None = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", self.on_save)
        self.root.bind("c", self.on_clear)
        self.root.bind("C", self.on_clear)
        self.root.bind("q", self.on_quit)
        self.root.bind("Q", self.on_quit)
        self.root.bind("<Escape>", self.on_quit)

    def to_image_xy(self, x: int, y: int) -> tuple[int, int]:
        ix = int(round(x / max(self.scale, 1e-6)))
        iy = int(round(y / max(self.scale, 1e-6)))
        return min(max(ix, 0), self.image.width - 1), min(max(iy, 0), self.image.height - 1)

    def to_canvas_xy(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def set_rect_from_canvas(self, x1c: int, y1c: int, x2c: int, y2c: int) -> None:
        x1, y1 = self.to_image_xy(x1c, y1c)
        x2, y2 = self.to_image_xy(x2c, y2c)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        if x2 - x1 < 20 or y2 - y1 < 20:
            self.rect = None
            self.status.set("Selection too small. Drag a larger rectangle.")
            return
        self.rect = (x1, y1, x2, y2)
        self.status.set(f"Selected x={x1} y={y1} w={x2 - x1} h={y2 - y1}. Enter saves, C clears.")

    def on_press(self, event: tk.Event) -> None:
        self.start = (int(event.x), int(event.y))
        if self.rect_canvas_id is not None:
            self.canvas.delete(self.rect_canvas_id)
        self.rect_canvas_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="cyan", width=3)

    def on_drag(self, event: tk.Event) -> None:
        if self.start is None or self.rect_canvas_id is None:
            return
        self.canvas.coords(self.rect_canvas_id, self.start[0], self.start[1], event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        if self.start is None:
            return
        self.set_rect_from_canvas(self.start[0], self.start[1], int(event.x), int(event.y))

    def on_clear(self, _event: tk.Event | None = None) -> None:
        self.rect = None
        self.start = None
        if self.rect_canvas_id is not None:
            self.canvas.delete(self.rect_canvas_id)
            self.rect_canvas_id = None
        self.status.set("Cleared. Drag rectangle. Enter saves. C clears. Q/Esc exits.")

    def on_save(self, _event: tk.Event | None = None) -> None:
        if self.rect is None:
            self.status.set("No valid rectangle selected yet.")
            return
        out_dir = save_image_selection(self.image, self.image_path, self.rect, self.output_dir)
        self.status.set(f"Saved: {out_dir}. Drag another rectangle or Q/Esc to exit.")
        print(f"Saved selection: {out_dir}")
        self.on_clear()

    def on_quit(self, _event: tk.Event | None = None) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively select review-only rectangular crops")
    parser.add_argument("--image", type=Path, required=True, help="Source PNG/JPG image to crop directly")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ruler_manual_annotation_v1/review/manual_rect_selection"),
        help="Directory where timestamped crop selections are written",
    )
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=850)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    CropSelector(args.image, args.output_dir, args.max_width, args.max_height).run()


if __name__ == "__main__":
    main()
