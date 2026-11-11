from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "reports/ruler_v7/clean_eval"
MANIFEST = REVIEW_DIR / "possible_natural_edge_contact_manifest.csv"
DECISIONS = REVIEW_DIR / "review_decisions.csv"
SUMMARY = REVIEW_DIR / "review_summary.csv"
DATASET = ROOT / "data/trunk_dataset_combined_v7_clean_eval/yolo_seg"

DECISION_VALUES = ("KEEP_COMPLETE", "EXCLUDE_TRUNCATED", "UNSURE")


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_decisions() -> dict[str, dict[str, str]]:
    if not DECISIONS.exists():
        return {}
    with DECISIONS.open(newline="", encoding="utf-8") as f:
        return {row["image_name"]: row for row in csv.DictReader(f)}


def write_decisions(rows: list[dict[str, str]], decisions: dict[str, dict[str, str]]) -> None:
    fieldnames = [
        "image_name",
        "decision",
        "reviewed_at",
        "source_id",
        "source_image",
        "detection_id",
        "label_source",
        "manual_status",
        "notes",
    ]
    tmp = DECISIONS.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            existing = decisions.get(row["image_name"])
            if existing:
                writer.writerow({key: existing.get(key, "") for key in fieldnames})
    tmp.replace(DECISIONS)


def write_summary(rows: list[dict[str, str]], decisions: dict[str, dict[str, str]]) -> None:
    counts = Counter(d.get("decision", "") for d in decisions.values() if d.get("decision"))
    pending = len(rows) - sum(counts.values())
    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decision", "count"])
        writer.writeheader()
        for decision in DECISION_VALUES:
            writer.writerow({"decision": decision, "count": counts.get(decision, 0)})
        writer.writerow({"decision": "PENDING", "count": pending})
        writer.writerow({"decision": "TOTAL", "count": len(rows)})


def yolo_polygons(label_path: Path, width: int, height: int) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        values = [float(v) for v in parts[1:]]
        polygons.append([(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)])
    return polygons


def render_overlay(row: dict[str, str], max_w: int, max_h: int) -> Image.Image:
    image_path = DATASET / "images/val" / row["image_name"]
    label_path = DATASET / "labels/val" / row["label_name"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for polygon in yolo_polygons(label_path, image.width, image.height):
        draw.polygon(polygon, fill=(0, 80, 255, 95), outline=(255, 255, 0, 255))
        draw.line(polygon + [polygon[0]], fill=(255, 255, 0, 255), width=2)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 0, 255), width=2)
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    return image


class ReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.rows = load_manifest()
        self.decisions = load_decisions()
        self.index = self.first_pending_index()
        self.photo: ImageTk.PhotoImage | None = None

        root.title("Possible Natural Edge Contact Review")
        root.geometry("1100x850")
        root.bind("<Left>", lambda _event: self.prev_image())
        root.bind("<Right>", lambda _event: self.next_image())
        root.bind("k", lambda _event: self.set_decision("KEEP_COMPLETE"))
        root.bind("e", lambda _event: self.set_decision("EXCLUDE_TRUNCATED"))
        root.bind("u", lambda _event: self.set_decision("UNSURE"))
        root.bind("s", lambda _event: self.save())

        self.header = tk.Label(root, text="", font=("Segoe UI", 12), justify="left")
        self.header.pack(fill="x", padx=8, pady=4)
        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(expand=True, fill="both", padx=8, pady=4)
        self.info = tk.Label(root, text="", font=("Segoe UI", 10), justify="left")
        self.info.pack(fill="x", padx=8, pady=4)

        buttons = tk.Frame(root)
        buttons.pack(fill="x", padx=8, pady=8)
        tk.Button(buttons, text="Prev [Left]", command=self.prev_image).pack(side="left")
        tk.Button(buttons, text="KEEP_COMPLETE [k]", command=lambda: self.set_decision("KEEP_COMPLETE"), bg="#d8ffd8").pack(side="left", padx=8)
        tk.Button(buttons, text="EXCLUDE_TRUNCATED [e]", command=lambda: self.set_decision("EXCLUDE_TRUNCATED"), bg="#ffd8d8").pack(side="left", padx=8)
        tk.Button(buttons, text="UNSURE [u]", command=lambda: self.set_decision("UNSURE"), bg="#fff4c2").pack(side="left", padx=8)
        tk.Button(buttons, text="Next [Right]", command=self.next_image).pack(side="left", padx=8)
        tk.Button(buttons, text="Save Summary [s]", command=self.save).pack(side="right")

        self.refresh()

    def first_pending_index(self) -> int:
        for idx, row in enumerate(self.rows):
            if row["image_name"] not in self.decisions:
                return idx
        return 0

    def current_row(self) -> dict[str, str]:
        return self.rows[self.index]

    def refresh(self) -> None:
        row = self.current_row()
        decision = self.decisions.get(row["image_name"], {}).get("decision", "PENDING")
        counts = Counter(d.get("decision", "") for d in self.decisions.values() if d.get("decision"))
        reviewed = sum(counts.values())
        self.header.config(
            text=(
                f"{self.index + 1}/{len(self.rows)} | reviewed {reviewed}/{len(self.rows)} | current: {decision}\n"
                f"{row['image_name']} | source_id={row['source_id']} | detection={row['detection_id']}"
            )
        )
        image = render_overlay(row, max_w=1040, max_h=650)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)
        self.info.config(
            text=(
                "Yellow/blue = YOLO polygon from label, magenta = image boundary.\n"
                f"source_image={row['source_image']}\n"
                f"label_source={row['label_source']} | manual_status={row['manual_status']} | source_crop={row['source_crop']}\n"
                f"Autosave: {DECISIONS.relative_to(ROOT)}"
            )
        )

    def set_decision(self, decision: str) -> None:
        row = self.current_row()
        self.decisions[row["image_name"]] = {
            "image_name": row["image_name"],
            "decision": decision,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "source_id": row["source_id"],
            "source_image": row["source_image"],
            "detection_id": row["detection_id"],
            "label_source": row["label_source"],
            "manual_status": row["manual_status"],
            "notes": "",
        }
        self.save(show_message=False)
        if self.index < len(self.rows) - 1:
            self.index += 1
        self.refresh()

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def next_image(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.refresh()

    def save(self, show_message: bool = True) -> None:
        write_decisions(self.rows, self.decisions)
        write_summary(self.rows, self.decisions)
        if show_message:
            messagebox.showinfo("Saved", f"Saved decisions and summary.\n{DECISIONS}\n{SUMMARY}")


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
