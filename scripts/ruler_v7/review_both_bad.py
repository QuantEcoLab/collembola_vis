"""Review causes of existing R/BOTH_BAD decisions, without model execution.

Default: a deterministic, source-balanced sample of 15. Use --all for all
eligible cases. Only both_bad_reason_review.csv and both_bad_reason_summary.csv
are written; the original position and model-rating CSVs are read-only inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox

sys.dont_write_bytecode = True
from PIL import Image, ImageDraw, ImageFont, ImageTk
from manual_contour_review import HERE, ROOT, input_path, key, read_csv, render, review_font, write_csv

REASONS = {
    "1": "CROP_CUTS_BODY",
    "2": "RULER_INCLUDED",
    "3": "BODY_PART_MISSING",
    "4": "MULTIPLE_PROBLEMS",
    "5": "UNCLEAR",
}
LABELS = {
    "1": "1 - crop je odrezao trup",
    "2": "2 - trup cijeli, maska zahvaca ravnalo",
    "3": "3 - trup cijeli, maska izostavlja dio trupa",
    "4": "4 - vise problema istodobno",
    "5": "5 - nejasno / ne mogu odrediti",
}
FIELDS = ["source_id", "detection_id", "reason", "reviewed_at", "notes",
          "position_reviewed_at", "model_decision_reviewed_at"]


def load_inputs():
    positions = {key(row): row for row in read_csv(HERE / "ruler_position_review.csv")}
    decisions = {key(row): row for row in read_csv(HERE / "manual_review_decisions.csv")}
    rows = []
    for row in read_csv(HERE / "index.csv"):
        position, decision = positions.get(key(row), {}), decisions.get(key(row), {})
        if position.get("position_class") == "R" and decision.get("decision") == "BOTH_BAD":
            rows.append({**row, "position_reviewed_at": position["reviewed_at"],
                         "model_decision_reviewed_at": decision["reviewed_at"]})
    sources = {row["source_id"]: input_path(row["image_path"])
               for row in read_csv(HERE.parent / "unseen/candidate_selection_leakage_audit.csv")}
    for row in rows:
        if row["source_id"] not in sources or not sources[row["source_id"]].is_file():
            raise ValueError(f"Missing full-resolution source for {key(row)}")
        with Image.open(input_path(row["original_path"])) as crop:
            for name in ("v5", "v7"):
                if row[f"{name}_mask_path"]:
                    with Image.open(input_path(row[f"{name}_mask_path"])) as mask:
                        if mask.size != crop.size:
                            raise ValueError(f"Mask size mismatch: {key(row)} {name}")
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in review inputs")
    return rows, sources


def sample_cases(rows, limit):
    groups = {source: sorted([row for row in rows if row["source_id"] == source],
                             key=lambda row: int(row["detection_id"]))
              for source in sorted({row["source_id"] for row in rows})}
    limit = min(limit, len(rows))
    allocation = Counter()
    while sum(allocation.values()) < limit:
        for source, group in groups.items():
            if allocation[source] < len(group) and sum(allocation.values()) < limit:
                allocation[source] += 1
    selected = []
    for source, group in groups.items():
        count = allocation[source]
        # Spread selection over detection IDs, rather than favoring the first IDs.
        for i in range(count):
            idx = round(i * (len(group) - 1) / (count - 1)) if count > 1 else len(group) // 2
            selected.append(group[idx])
    return selected


class ReasonState:
    def __init__(self, all_rows, queue, directory=HERE):
        self.all_rows, self.queue, self.directory = all_rows, queue, directory
        self.records = {}
        for row in read_csv(directory / "both_bad_reason_review.csv"):
            if key(row) in self.records or row.get("reason") not in REASONS.values():
                raise ValueError("Invalid or duplicate reason review record")
            self.records[key(row)] = row

    def reviewed(self, row):
        saved = self.records.get(key(row), {})
        return (saved.get("reason") in REASONS.values()
                and saved.get("position_reviewed_at") == row["position_reviewed_at"]
                and saved.get("model_decision_reviewed_at") == row["model_decision_reviewed_at"])

    def pending(self):
        return next((i for i, row in enumerate(self.queue) if not self.reviewed(row)), None)

    def record(self, row, choice, notes):
        identity = key(row)
        self.records[identity] = dict(source_id=identity[0], detection_id=identity[1],
                                      reason=REASONS[choice], notes=notes,
                                      reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      position_reviewed_at=row["position_reviewed_at"],
                                      model_decision_reviewed_at=row["model_decision_reviewed_at"])
        self.save()

    def summary(self):
        fields = ["scope", "source_id", "eligible_total", "selected_count", "reviewed_count",
                  "pending_count", *REASONS.values()]
        result = []
        for source in ["ALL", *sorted({row["source_id"] for row in self.queue})]:
            subset = [row for row in self.queue if source == "ALL" or row["source_id"] == source]
            counts = Counter(self.records[key(row)]["reason"] for row in subset if self.reviewed(row))
            result.append(dict(scope="sample" if len(self.queue) < len(self.all_rows) else "all_eligible",
                               source_id=source,
                               eligible_total=sum(source == "ALL" or row["source_id"] == source for row in self.all_rows),
                               selected_count=len(subset), reviewed_count=sum(counts.values()),
                               pending_count=len(subset) - sum(counts.values()),
                               **{reason: counts[reason] for reason in REASONS.values()}))
        return fields, result

    def save(self):
        # Preserve earlier reviews outside the current sample or eligibility set.
        write_csv(self.directory / "both_bad_reason_review.csv", FIELDS,
                  [self.records[identity] for identity in sorted(self.records)])
        fields, rows = self.summary()
        write_csv(self.directory / "both_bad_reason_summary.csv", fields, rows)


class Context:
    def __init__(self, sources):
        self.sources, self.source, self.image = sources, None, None

    def close(self):
        if self.image is not None:
            self.image.close()
        self.image, self.source = None, None

    def render(self, row, width, height):
        source = row["source_id"]
        if source != self.source:
            self.close()
            # Only one full-resolution source is held in memory at a time.
            self.image = Image.open(self.sources[source])
            self.image.load()
            self.source = source
        x1, y1, x2, y2 = [float(row[f"bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2")]
        context_w, context_h = max(1000, 3 * (x2 - x1)), max(850, 3 * (y2 - y1))
        left = max(0, int((x1 + x2 - context_w) / 2))
        top = max(0, int((y1 + y2 - context_h) / 2))
        right, bottom = min(self.image.width, int(left + context_w)), min(self.image.height, int(top + context_h))
        crop = self.image.crop((left, top, right, bottom)).convert("RGB")
        scale = min((width - 20) / crop.width, (height - 48) / crop.height)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        draw.rectangle(((x1 - left) * scale, (y1 - top) * scale,
                        (x2 - left) * scale, (y2 - top) * scale), outline="red", width=3)
        canvas = Image.new("RGB", (width, height), "#ededed")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), "Siri kontekst - crveni okvir = crop koji su modeli dobili",
                  fill="black", font=review_font(20))
        canvas.paste(crop, ((width - crop.width) // 2, 42 + (height - 48 - crop.height) // 2))
        return canvas


class App:
    def __init__(self, root, state, sources):
        self.root, self.state = root, state
        self.context = Context(sources)
        self.index = state.pending()
        self.photos, self.resize_job = [], None
        root.title("Oba losa - razlog: crop ili segmentacija")
        root.geometry(f"1450x{min(960, root.winfo_screenheight() - 100)}")
        root.minsize(950, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<KeyRelease>", self.on_key)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        self.header = tk.Label(root, anchor="w", justify="left", font=("Segoe UI", 12))
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=6)
        self.guide = tk.Label(root, anchor="w", justify="left", wraplength=1400,
                              text="1 = nedostaje trup IZVAN crvenog okvira. 2/3 = cijeli trup je UNUTAR okvira, ali maska grijesi.\n"
                                   "4 = vise problema; opisi ih u biljesci. 5 = nejasno. Ne biras ponovno bolji model.")
        self.guide.grid(row=1, column=0, sticky="ew", padx=12)
        images = tk.Frame(root)
        images.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        images.columnconfigure(0, weight=1)
        self.images = []
        for i in range(2):
            images.rowconfigure(i, weight=1)
            frame = tk.Frame(images, bg="#ededed")
            frame.grid(row=i, column=0, sticky="nsew", pady=3)
            label = tk.Label(frame, bg="#ededed")
            label.place(x=0, y=0, relwidth=1, relheight=1)
            label.bind("<Configure>", self.on_resize)
            self.images.append(label)
        note_frame = tk.Frame(root)
        note_frame.grid(row=3, column=0, sticky="ew", padx=12)
        tk.Label(note_frame, text="Biljeska:").pack(side="left")
        self.notes = tk.Entry(note_frame)
        self.notes.pack(side="left", fill="x", expand=True, padx=6)
        self.notes.bind("<Escape>", lambda event: root.focus_set())
        self.buttons = []
        choices = tk.Frame(root)
        choices.grid(row=4, column=0, sticky="w", padx=12, pady=6)
        for i, (choice, text) in enumerate(LABELS.items()):
            button = tk.Button(choices, text=text, command=lambda choice=choice: self.choose(choice))
            button.grid(row=i // 3, column=i % 3, sticky="w", padx=3, pady=2)
            self.buttons.append(button)
        nav = tk.Frame(root)
        nav.grid(row=5, column=0, sticky="ew", padx=12, pady=6)
        tk.Button(nav, text="B - prethodni", command=self.previous).pack(side="left")
        tk.Button(nav, text="Sazetak", command=self.show_summary).pack(side="left", padx=8)
        tk.Button(nav, text="Spremi i zatvori", command=self.close).pack(side="right")
        self.refresh()

    def current(self):
        return self.state.queue[self.index] if self.index is not None else None

    def on_resize(self, event):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(150, self.redraw)

    def redraw(self):
        self.resize_job = None
        row = self.current()
        if row is None:
            for label in self.images:
                label.config(image="", text="Odabrani pregled dovrsen. Sazetak ili B za povratak.")
            self.photos = []
            return
        try:
            context = self.context.render(row, max(900, self.images[0].winfo_width()), max(160, self.images[0].winfo_height()))
            triptych = render(row, "rating", max(900, self.images[1].winfo_width()), max(160, self.images[1].winfo_height()))
            self.photos = [ImageTk.PhotoImage(image) for image in (context, triptych)]
            for label, photo in zip(self.images, self.photos):
                label.config(image=photo, text="")
        except Exception as exc:
            messagebox.showerror("Greska citanja", str(exc))

    def refresh(self):
        row = self.current()
        saved = self.state.records.get(key(row), {}) if row else {}
        reviewed = sum(self.state.reviewed(row) for row in self.state.queue)
        identity = f"{row['source_id']} | detection {row['detection_id']}" if row else "DOVRSENO"
        self.header.config(text=f"Oba losa: pregledano {reviewed}/{len(self.state.queue)} od ukupno {len(self.state.all_rows)}\n"
                           f"{identity} | razlog: {saved.get('reason', 'PENDING')}")
        self.notes.delete(0, "end")
        self.notes.insert(0, saved.get("notes", ""))
        for button in self.buttons:
            button.config(state="normal" if row else "disabled")
        self.redraw()

    def save_notes(self):
        row = self.current()
        if row and self.state.reviewed(row):
            self.state.records[key(row)]["notes"] = self.notes.get()
            self.state.save()

    def choose(self, choice):
        row = self.current()
        if row is None:
            return
        try:
            self.state.record(row, choice, self.notes.get())
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        self.index = self.index + 1 if self.index + 1 < len(self.state.queue) else self.state.pending()
        self.root.focus_set()
        self.refresh()
        if self.index is None:
            self.show_summary()

    def previous(self):
        try:
            self.save_notes()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        if self.index is None:
            self.index = len(self.state.queue) - 1 if self.state.queue else None
        elif self.index > 0:
            self.index -= 1
        self.root.focus_set()
        self.refresh()

    def on_key(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        choice = event.keysym.upper()
        if choice == "B":
            self.previous()
        elif choice in REASONS:
            self.choose(choice)

    def show_summary(self):
        try:
            self.save_notes()
            self.state.save()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        window = tk.Toplevel(self.root)
        window.title("Razlozi - samo pregledani uzorak, bez izbora modela")
        text = tk.Text(window, width=95, height=30, wrap="word", font=("Consolas", 11))
        text.pack(fill="both", expand=True)
        for row in self.state.summary()[1]:
            text.insert("end", f"{row['source_id']}: pregledano {row['reviewed_count']}/{row['selected_count']}\n")
            for choice, label in LABELS.items():
                text.insert("end", f"  {label}: {row[REASONS[choice]]}\n")
            text.insert("end", "\n")
        text.insert("end", "Ovo je pregled razloga u odabranom uzorku; nije mjera kvalitete cijelog dataseta.\n")
        text.config(state="disabled")

    def close(self):
        try:
            self.save_notes()
            self.state.save()
        except Exception as exc:
            messagebox.showerror("Spremanje nije uspjelo", str(exc))
            return
        self.context.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Review all current R/BOTH_BAD cases instead of the default 15")
    parser.add_argument("--check", action="store_true", help="Validate inputs and print selection without GUI or writes")
    args = parser.parse_args()
    rows, sources = load_inputs()
    if not rows:
        raise SystemExit("No R/BOTH_BAD cases found in the saved reviews")
    queue = sorted(rows, key=key) if args.all else sample_cases(rows, 15)
    state = ReasonState(rows, queue)
    if args.check:
        print(f"OK: {len(rows)} R/BOTH_BAD cases; selected {len(queue)}; next unreviewed index: {state.pending()}")
        print(dict(Counter(row["source_id"] for row in queue)))
        return
    root = tk.Tk()
    App(root, state, sources)
    root.mainloop()


if __name__ == "__main__":
    main()
