"""Two-stage manual review of existing crops/masks, with no model execution.

Run without arguments for the GUI, --check to validate inputs without writing,
or --summary to display/export the current R-only summary as a CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "reports/ruler_v7/manual_review"
POSITION_FIELDS = ["source_id", "detection_id", "position_class", "reviewed_at", "notes"]
DECISION_FIELDS = ["source_id", "detection_id", "decision", "reviewed_at", "notes"]
CHOICES = {
    "1": "V5_BETTER", "2": "V7_BETTER", "3": "EQUAL",
    "4": "BOTH_BAD", "5": "UNASSESSABLE",
}
POSITION_LABELS = {
    "R": "R - trup dodiruje/prekriva crtu ili brojku",
    "N": "N - jedinka uz ravnalo, bez presijecanja",
    "X": "X - nije jedinka ili nije uz ravnalo",
    "U": "U - nejasno / preskoci",
}
RATING_LABELS = {
    "1": "1 - V5 bolji", "2": "2 - V7 bolji", "3": "3 - jednaki",
    "4": "4 - oba losa", "5": "5 - nije moguce ocijeniti",
}


def key(row):
    return row["source_id"], str(int(row["detection_id"]))


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    temporary = path.with_name(path.stem + ".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def input_path(value):
    path = (ROOT / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Input outside project: {value}")
    return path


def review_font(size):
    """Use Windows Arial when available, then Linux fonts, then Pillow's default."""
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


def load_cases():
    rows = [row for row in read_csv(HERE / "index.csv")
            if row.get("difference_review_candidate", "").lower() == "true"]
    if len(rows) != 130 or len({key(row) for row in rows}) != 130:
        raise ValueError("Expected exactly 130 unique priority cases in index.csv")
    for row in rows:
        with Image.open(input_path(row["original_path"])) as original:
            for name in ("v5", "v7"):
                value = row[f"{name}_mask_path"]
                expected = row[f"{name}_has_mask"].lower() == "true"
                if bool(value) != expected:
                    raise ValueError(f"Inconsistent mask metadata: {key(row)} {name}")
                if value:
                    with Image.open(input_path(value)) as mask:
                        if mask.size != original.size:
                            raise ValueError(f"Mask size mismatch: {key(row)} {name}")
    return sorted(rows, key=lambda row: (
        -int(bool(row["v5_mask_path"]) != bool(row["v7_mask_path"])),
        float(row["mask_pixel_iou_for_sorting"] or -1),
    ))


class ReviewState:
    def __init__(self, rows, directory=HERE):
        self.rows = rows
        self.directory = directory
        self.positions = self.load("ruler_position_review.csv", "position_class", set(POSITION_LABELS))
        self.decisions = self.load("manual_review_decisions.csv", "decision", set(CHOICES.values()))

    def load(self, filename, column, allowed):
        records = read_csv(self.directory / filename)
        result = {}
        valid = {key(row) for row in self.rows}
        for row in records:
            identity = key(row)
            if identity not in valid or identity in result or row.get(column) not in allowed:
                raise ValueError(f"Invalid/duplicate/unknown record in {filename}: {row}")
            result[identity] = row
        return result

    def position_done(self):
        return all(key(row) in self.positions for row in self.rows)

    def ruler_rows(self):
        return [row for row in self.rows if self.positions.get(key(row), {}).get("position_class") == "R"]

    def pending(self, phase):
        rows = self.rows if phase == "position" else self.ruler_rows()
        records = self.positions if phase == "position" else self.decisions
        return next((i for i, row in enumerate(rows) if key(row) not in records), None)

    def record(self, row, phase, value, notes):
        identity = key(row)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if phase == "position":
            if value not in POSITION_LABELS:
                raise ValueError(value)
            previous = self.positions.get(identity, {}).get("position_class")
            self.positions[identity] = dict(source_id=identity[0], detection_id=identity[1],
                                            position_class=value, reviewed_at=timestamp, notes=notes)
            if previous != value:
                self.decisions.pop(identity, None)
        else:
            if not self.position_done() or row not in self.ruler_rows() or value not in CHOICES.values():
                raise ValueError("Contour rating requires completed position phase and class R")
            self.decisions[identity] = dict(source_id=identity[0], detection_id=identity[1],
                                            decision=value, reviewed_at=timestamp, notes=notes)
        self.save()

    def summary(self):
        fields = ["scope", "source_id", "priority_cases", "R", "N", "X", "U", "position_pending",
                  "ruler_individuals", *CHOICES.values(), "rating_pending", "both_masks",
                  "v5_only_mask", "v7_only_mask", "one_model_mask", "neither_mask"]
        summaries = []
        for source in ["ALL", *sorted({row["source_id"] for row in self.rows})]:
            subset = [row for row in self.rows if source == "ALL" or row["source_id"] == source]
            positions = Counter(self.positions.get(key(row), {}).get("position_class") for row in subset)
            rulers = [row for row in subset if self.positions.get(key(row), {}).get("position_class") == "R"]
            decisions = Counter(self.decisions.get(key(row), {}).get("decision") for row in rulers)
            only5 = sum(bool(row["v5_mask_path"]) and not row["v7_mask_path"] for row in rulers)
            only7 = sum(bool(row["v7_mask_path"]) and not row["v5_mask_path"] for row in rulers)
            summaries.append(dict(
                scope="overall" if source == "ALL" else "source", source_id=source,
                priority_cases=len(subset), R=positions["R"], N=positions["N"], X=positions["X"], U=positions["U"],
                position_pending=positions[None], ruler_individuals=len(rulers),
                **{value: decisions[value] for value in CHOICES.values()},
                rating_pending=decisions[None],
                both_masks=sum(bool(row["v5_mask_path"]) and bool(row["v7_mask_path"]) for row in rulers),
                v5_only_mask=only5, v7_only_mask=only7, one_model_mask=only5 + only7,
                neither_mask=sum(not row["v5_mask_path"] and not row["v7_mask_path"] for row in rulers),
            ))
        return fields, summaries

    def save(self):
        for filename, fields, records in (
            ("ruler_position_review.csv", POSITION_FIELDS, self.positions),
            ("manual_review_decisions.csv", DECISION_FIELDS, self.decisions),
        ):
            write_csv(self.directory / filename, fields,
                      [records[key(row)] for row in self.rows if key(row) in records])
        self.save_summary()

    def save_summary(self):
        fields, summaries = self.summary()
        write_csv(self.directory / "manual_review_summary.csv", fields, summaries)


def render(row, phase, width, height):
    with Image.open(input_path(row["original_path"])) as image:
        original = np.array(image.convert("RGB"))
    panels = 1 if phase == "position" else 3
    panel_width = max(80, width // panels)
    scale = min((panel_width - 20) / original.shape[1], (height - 60) / original.shape[0])
    size = (max(1, round(original.shape[1] * scale)), max(1, round(original.shape[0] * scale)))
    enlarged = cv2.resize(original, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    output = Image.new("RGB", (panel_width * panels, height), "#ededed")
    draw = ImageDraw.Draw(output)
    title_font = review_font(24)
    for i, label in enumerate(("original", "v5", "v7")[:panels]):
        panel = enlarged.copy()
        no_mask = False
        if i:
            value = row[f"{label}_mask_path"]
            no_mask = not value
            if value:
                with Image.open(input_path(value)) as image:
                    mask = np.array(image.convert("L")) > 0
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [np.round(contour.astype(float) * scale).astype(np.int32) for contour in contours]
                cv2.drawContours(panel, contours, -1, (255, 215, 0), 2, cv2.LINE_AA)
        draw.text((i * panel_width + 10, 8), label + (" - NO MASK" if no_mask else ""), fill="black", font=title_font)
        output.paste(Image.fromarray(panel), (i * panel_width + (panel_width - size[0]) // 2,
                                            48 + (height - 60 - size[1]) // 2))
    return output


class ReviewApp:
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.phase = "rating" if state.position_done() else "position"
        self.index = state.pending(self.phase)
        self.photo = None
        self.resize_job = None
        root.title("Ruler Review - raw v5 / v7")
        root.geometry("1500x930")
        root.minsize(950, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<KeyRelease>", self.on_key)
        self.header = tk.Label(root, anchor="w", justify="left", font=("Segoe UI", 12))
        self.header.pack(fill="x", padx=12, pady=5)
        self.guide = tk.Label(root, anchor="w", justify="left", wraplength=1450, font=("Segoe UI", 11))
        self.guide.pack(fill="x", padx=12)
        root.bind("<Configure>", lambda event: self.guide.config(wraplength=max(400, event.width - 40))
                  if event.widget == root else None)
        image_frame = tk.Frame(root, bg="#ededed")
        image_frame.pack(fill="both", expand=True, padx=12, pady=6)
        # An image must not increase the requested window size on every redraw.
        self.image = tk.Label(image_frame, bg="#ededed")
        self.image.place(x=0, y=0, relwidth=1, relheight=1)
        self.image.bind("<Configure>", self.on_resize)
        notes_frame = tk.Frame(root)
        notes_frame.pack(fill="x", padx=12)
        tk.Label(notes_frame, text="Biljeske (neobavezno):").pack(side="left")
        self.notes = tk.Entry(notes_frame, font=("Segoe UI", 11))
        self.notes.pack(side="left", fill="x", expand=True, padx=8)
        self.notes.bind("<Escape>", lambda event: root.focus_set())
        self.actions = tk.Frame(root)
        self.actions.pack(fill="x", padx=12, pady=8)
        nav = tk.Frame(root)
        nav.pack(fill="x", padx=12, pady=4)
        tk.Button(nav, text="B - prethodni", command=self.previous).pack(side="left")
        tk.Button(nav, text="Pregled polozaja", command=lambda: self.switch("position")).pack(side="left", padx=8)
        tk.Button(nav, text="Samo R - usporedba", command=lambda: self.switch("rating")).pack(side="left")
        tk.Button(nav, text="Sazetak", command=self.show_summary).pack(side="right")
        tk.Button(nav, text="Spremi i zatvori", command=self.close).pack(side="right", padx=8)
        self.status = tk.Label(root, anchor="w", text="Svaka odluka se odmah sprema. Esc izlazi iz unosa biljeski.")
        self.status.pack(fill="x", padx=12, pady=4)
        self.refresh()

    def cases(self):
        return self.state.rows if self.phase == "position" else self.state.ruler_rows()

    def current(self):
        rows = self.cases()
        return rows[self.index] if self.index is not None and self.index < len(rows) else None

    def records(self):
        return self.state.positions if self.phase == "position" else self.state.decisions

    def on_resize(self, event):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(180, self.redraw)

    def redraw(self):
        self.resize_job = None
        row = self.current()
        if row is None:
            self.photo = None
            self.image.config(image="", text="Faza dovrsena. Otvori Sazetak ili se vrati na prethodni slucaj.")
            return
        try:
            image = render(row, self.phase, max(900, self.image.winfo_width()), max(260, self.image.winfo_height()))
            self.photo = ImageTk.PhotoImage(image)
            self.image.config(image=self.photo, text="")
        except Exception as exc:
            messagebox.showerror("Greska citanja", str(exc))

    def refresh(self):
        row, rows = self.current(), self.cases()
        field = "position_class" if self.phase == "position" else "decision"
        done = sum(key(item) in self.records() for item in rows)
        title = "1. Potvrda polozaja" if self.phase == "position" else "2. Raw konture - samo R"
        identity = f"{row['source_id']} | detection {row['detection_id']}" if row else "DOVRSENO"
        saved = self.records().get(key(row), {}) if row else {}
        self.header.config(text=f"{title} | ocijenjeno {done}/{len(rows)}\n{identity} | spremljeno: {saved.get(field, 'PENDING')}")
        self.guide.config(text=(
            "R samo ako dio TRUPA dodiruje, prekriva ili se vizualno spaja s bijelom vodoravnom crtom, "
            "okomitom oznakom ili brojkom/svijetlim tiskom. Sama blizina nije R. "
            "N = jedinka uz ravnalo bez kontakta; X = nije jedinka ili nije uz ravnalo. "
            "U = iz cropa nije jasno, spremi kao nejasno i predji dalje (ne ulazi u usporedbu)."
            if self.phase == "position" else
            "Usporedi cijeli trup, zahvacanje crta/brojki i spajanje s pozadinom. "
            "Isti crop i zoom za oba modela. NO MASK = model nema spremljenu raw masku. "
            "5 = slucaj nije moguce ocijeniti, ukljucujuci naknadno utvrdjenu neprikladnost."
        ))
        self.notes.delete(0, "end")
        self.notes.insert(0, saved.get("notes", ""))
        for child in self.actions.winfo_children():
            child.destroy()
        labels = POSITION_LABELS if self.phase == "position" else RATING_LABELS
        columns = 2 if self.phase == "position" else 3
        for i, (value, label) in enumerate(labels.items()):
            tk.Button(self.actions, text=label, command=lambda value=value: self.choose(value),
                       state="normal" if row else "disabled").grid(row=i // columns, column=i % columns,
                                                                    sticky="w", padx=4, pady=2)
        self.redraw()

    def save_notes(self):
        row = self.current()
        if row and key(row) in self.records():
            record = self.records()[key(row)]
            if record.get("notes", "") != self.notes.get():
                record["notes"] = self.notes.get()
                self.state.save()

    def choose(self, value):
        row = self.current()
        if row is None:
            return
        try:
            self.state.record(row, self.phase, value if self.phase == "position" else CHOICES[value], self.notes.get())
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        if self.phase == "position" and self.state.position_done():
            self.phase = "rating"
            self.index = self.state.pending("rating")
        elif self.index + 1 < len(self.cases()):
            self.index += 1
        else:
            self.index = self.state.pending(self.phase)
        self.root.focus_set()
        self.refresh()
        if self.phase == "rating" and self.index is None:
            self.show_summary()

    def previous(self):
        try:
            self.save_notes()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        if self.index is None:
            self.index = len(self.cases()) - 1 if self.cases() else None
        elif self.index > 0:
            self.index -= 1
        self.root.focus_set()
        self.refresh()

    def switch(self, phase):
        if phase == "rating" and not self.state.position_done():
            messagebox.showinfo("Prvo polozaj", "Najprije oznaci svih 130 slucajeva s R, N, X ili U (nejasno).")
            return
        try:
            self.save_notes()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        self.phase = phase
        self.index = self.state.pending(phase)
        if self.index is None and self.cases():
            self.index = 0
        self.refresh()

    def on_key(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        value = event.keysym.upper()
        if value == "B":
            self.previous()
        elif self.phase == "position" and value in POSITION_LABELS:
            self.choose(value)
        elif self.phase == "rating" and value in CHOICES:
            self.choose(value)

    def show_summary(self):
        try:
            self.save_notes()
            self.state.save_summary()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        _, rows = self.state.summary()
        window = tk.Toplevel(self.root)
        window.title("R-only sazetak - bez izbora production modela")
        text = tk.Text(window, width=105, height=32, font=("Consolas", 11), wrap="word")
        text.pack(fill="both", expand=True)
        for row in rows:
            text.insert("end", f"{row['source_id']}\n"
                        f"Polozaj: R={row['R']}  N={row['N']}  X={row['X']}  U/nejasno={row['U']}  neocijenjeno={row['position_pending']}\n"
                        f"Stvarne ruler jedinke (R): {row['ruler_individuals']}\n"
                        f"V5 bolji: {row['V5_BETTER']} | V7 bolji: {row['V7_BETTER']} | jednaki: {row['EQUAL']}\n"
                        f"Oba losa: {row['BOTH_BAD']} | nije moguce ocijeniti: {row['UNASSESSABLE']} | ceka ocjenu: {row['rating_pending']}\n"
                        f"Obje maske: {row['both_masks']} | samo v5: {row['v5_only_mask']} | samo v7: {row['v7_only_mask']} | bez maske: {row['neither_mask']}\n\n")
        text.insert("end", "Sazetak se odnosi samo na odabranih 130 prioritetnih slucajeva, ne na svih 779 detekcija.\n")
        text.config(state="disabled")

    def close(self):
        try:
            self.save_notes()
            self.state.save()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only input/decision validation; no GUI or CSV writes")
    parser.add_argument("--summary", action="store_true", help="Export current summary CSV and print counts; no GUI")
    args = parser.parse_args()
    state = ReviewState(load_cases())
    if args.check:
        print(f"OK: {len(state.rows)} priority cases, {len(state.positions)} position reviews, {len(state.decisions)} ratings")
        print(f"R-only queue: {len(state.ruler_rows())}; next position: {state.pending('position')}; next R rating: {state.pending('rating')}")
        return
    if args.summary:
        state.save_summary()
        for row in state.summary()[1]:
            print(row)
        return
    root = tk.Tk()
    ReviewApp(root, state)
    root.mainloop()


if __name__ == "__main__":
    main()
