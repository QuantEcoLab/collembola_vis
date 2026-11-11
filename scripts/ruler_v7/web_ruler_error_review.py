#!/usr/bin/env python3
"""Remote-SSH review of existing ruler BOTH_BAD cases; no inference or Tk/OpenCV.

Reads raw assets directly from an external snapshot. New, per-model decisions
are append-only SQLite events in outputs/, never edits to the historical CSVs.
Only context thumbnails are encoded on demand in memory. Contours are drawn by
the browser from existing masks, without generating comparison files on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = Path(
    "/home/jana/collembola-rescue/incoming/"
    "ruler-v7-20261027T162453Z/snapshot/repo"
)
ERRORS = {
    "BODY_PART_MISSING": "Nedostaje dio tijela",
    "RULER_INCLUDED": "Maska uključuje crtu/broj ravnala",
    "TOO_LARGE": "Maska je prevelika",
    "TOO_SMALL": "Maska je premala",
    "MULTIPLE_PROBLEMS": "Više problema",
    "UNASSESSABLE": "Nije moguće procijeniti",
}
CROP_STATES = {
    "COMPLETE": "Tijelo je cijelo unutar cropa",
    "TRUNCATED": "Rub cropa odrezuje tijelo",
    "UNCLEAR": "Iz konteksta nije jasno",
}
MODELS = ("crop", "v5", "v7")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def identity(row):
    return row["source_id"], row["detection_id"]


class ReviewInputs:
    """Read-only catalog. Historical decisions are references, not new ratings."""

    def __init__(self, reports, assets, expected_total=779, expected_cases=57):
        self.reports, self.assets = reports.resolve(), assets.resolve()
        review = self.reports / "manual_review"
        names = [review / name for name in (
            "index.csv", "ruler_position_review.csv", "manual_review_decisions.csv",
            "both_bad_reason_review.csv",
        )]
        names.append(self.reports / "unseen/candidate_selection_leakage_audit.csv")
        self.fingerprint = hashlib.sha256(
            b"".join(hashlib.sha256(path.read_bytes()).digest() for path in names)
        ).hexdigest()
        self.index = read_csv(names[0])
        positions = {identity(row): row for row in read_csv(names[1])}
        ratings = {identity(row): row for row in read_csv(names[2])}
        prior = {identity(row): row for row in read_csv(names[3])}
        self.sources = {row["source_id"]: row for row in read_csv(names[4])}
        if len({identity(row) for row in self.index}) != len(self.index):
            raise ValueError("Duplicate IDs in raw index")
        if expected_total is not None and len(self.index) != expected_total:
            raise ValueError(f"Expected {expected_total} raw index entries")
        self.cases = [dict(row) for row in self.index
                      if positions.get(identity(row), {}).get("position_class") == "R"
                      and ratings.get(identity(row), {}).get("decision") == "BOTH_BAD"]
        if expected_cases is not None and len(self.cases) != expected_cases:
            raise ValueError(f"Expected {expected_cases} R/BOTH_BAD cases")
        # The 42 cases without an old reason first; the old 15 remain available
        # for the new per-model schema, without inferring ratings from old labels.
        self.cases.sort(key=lambda row: (identity(row) in prior,
                                        row["source_id"], int(row["detection_id"])))
        Image.MAX_IMAGE_PIXELS = None  # Trusted local microscope images.
        self.source_sizes = {}
        for source, row in self.sources.items():
            with Image.open(self.asset(row["image_path"])) as image:
                self.source_sizes[source] = image.size
        for row in self.cases:
            with Image.open(self.asset(row["original_path"])) as image:
                row["crop_size"] = image.size
            row["prior_reason"] = prior.get(identity(row))
            row["historical_position"] = positions[identity(row)]
            row["historical_rating"] = ratings[identity(row)]
            self.validate_masks(row)
        self.audit = {
            "raw_index_count": len(self.index), "both_bad_count": len(self.cases),
            "historical_position_counts": dict(Counter(r["position_class"] for r in positions.values())),
            "historical_rating_counts": dict(Counter(r["decision"] for r in ratings.values())),
            "historical_reason_count": len(prior),
            "historical_reason_remaining": len(self.cases) - sum(identity(r) in prior for r in self.cases),
            "input_fingerprint": self.fingerprint,
        }

    def asset(self, value):
        path = (self.assets / value.replace("\\", "/")).resolve()
        if not path.is_relative_to(self.assets):
            raise ValueError("Asset path outside snapshot")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def validate_masks(self, row):
        for model in ("v5", "v7"):
            value = row[model + "_mask_path"]
            if bool(value) != (row[model + "_has_mask"].lower() == "true"):
                raise ValueError(f"Inconsistent mask presence: {identity(row)} {model}")
            if value:
                with Image.open(self.asset(value)) as image:
                    if image.size != tuple(row["crop_size"]):
                        raise ValueError(f"Mask/crop dimensions differ: {identity(row)} {model}")

    def context_box(self, row, wide=False):
        iw, ih = self.source_sizes[row["source_id"]]
        # Match the recorded pipeline's floor/ceil crop, not a rounded bbox.
        x = max(0, math.floor(float(row["bbox_x1"])))
        y = max(0, math.floor(float(row["bbox_y1"])))
        w, h = row["crop_size"]
        factor = 2 if wide else 1
        cw, ch = min(iw, max(1000, 4 * w) * factor), min(ih, max(850, 4 * h) * factor)
        left = max(0, min(int(x + w / 2 - cw / 2), iw - cw))
        top = max(0, min(int(y + h / 2 - ch / 2), ih - ch))
        return {"box": [left, top, left + cw, top + ch],
                "crop_box": [x - left, y - top, x + w - left, y + h - top]}

    def public_cases(self):
        return [{**{k: row[k] for k in (
                    "source_id", "detection_id", "crop_size", "prior_reason",
                    "historical_position", "historical_rating")},
                 "index": i, "has_masks": {m: bool(row[m + "_mask_path"]) for m in ("v5", "v7")},
                 "contexts": [self.context_box(row), self.context_box(row, True)]}
                for i, row in enumerate(self.cases)]


class ReviewStore:
    """Append events in transactions; original timestamps and revisions survive."""

    def __init__(self, inputs, output):
        self.inputs, self.output = inputs, output.resolve()
        if self.output.is_relative_to(inputs.assets) or self.output.is_relative_to(inputs.reports):
            raise ValueError("New decisions must be separate from snapshot and historical reports")
        self.output.mkdir(parents=True, exist_ok=True)
        self.database = self.output / "review.sqlite3"
        self.lock = threading.RLock()
        with self.connection() as con:
            con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS events ("
                        "id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, detection_id TEXT NOT NULL, "
                        "model TEXT NOT NULL, category TEXT NOT NULL, reviewed_at TEXT NOT NULL)")
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('input_fingerprint', ?)", (inputs.fingerprint,))
            saved = con.execute("SELECT value FROM metadata WHERE key='input_fingerprint'").fetchone()[0]
            if saved != inputs.fingerprint:
                raise ValueError("Saved review belongs to different input tables; choose a separate output directory")
        self.export_summary()

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.database, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def latest(self, con=None):
        if con is None:
            with self.connection() as connection:
                return self.latest(connection)
        rows = con.execute("SELECT * FROM events WHERE id IN ("
                           "SELECT MAX(id) FROM events GROUP BY source_id, detection_id, model)").fetchall()
        return {(row["source_id"], row["detection_id"], row["model"]): dict(row) for row in rows}

    def state(self):
        latest = self.latest()
        decisions, complete = [], []
        for row in self.inputs.cases:
            choices = {model: latest.get((*identity(row), model)) for model in MODELS}
            decisions.append(choices)
            complete.append(all(choices.values()))
        return {"decisions": decisions, "completed": sum(complete), "total": len(complete),
                "next_pending": next((i for i, done in enumerate(complete) if not done), None)}

    def record(self, index, model, category):
        if type(index) is not int or not 0 <= index < len(self.inputs.cases):
            raise ValueError("Invalid case index")
        allowed = CROP_STATES if model == "crop" else ERRORS if model in ("v5", "v7") else {}
        if category not in allowed:
            raise ValueError("Invalid model/category")
        row = self.inputs.cases[index]
        with self.lock:
            with self.connection() as con:
                con.execute("BEGIN IMMEDIATE")
                previous = self.latest(con).get((*identity(row), model))
                if previous is None or previous["category"] != category:
                    con.execute("INSERT INTO events (source_id,detection_id,model,category,reviewed_at) "
                                "VALUES (?,?,?,?,?)", (*identity(row), model, category,
                                 datetime.now(timezone.utc).isoformat(timespec="microseconds")))
            # The event is already durable if a derived export subsequently fails.
            try:
                self.export_summary()
            except OSError as exc:
                return {"state": self.state(), "export_warning": str(exc)}
            return {"state": self.state()}

    def summary(self):
        latest = self.latest()
        grouped = Counter()
        crop_counts = Counter()
        complete = 0
        for row in self.inputs.cases:
            source, detection = identity(row)
            complete += all((source, detection, model) in latest for model in MODELS)
            crop = latest.get((source, detection, "crop"), {}).get("category", "PENDING")
            crop_counts[crop] += 1
            for model in ("v5", "v7"):
                category = latest.get((source, detection, model), {}).get("category", "PENDING")
                grouped[(source, model, crop, category)] += 1
        rows = [{"source_id": s, "model": m, "crop_condition": c, "error_type": e, "count": n}
                for (s, m, c, e), n in sorted(grouped.items())]
        # Separate framing evidence from segmentation errors on a fully visible
        # body; missing anatomy in a truncated input alone cannot justify training.
        complete_body_errors = sum(n for (_s, _m, c, e), n in grouped.items()
                                   if c == "COMPLETE" and e not in {"PENDING", "UNASSESSABLE"})
        excess_mask_errors = sum(n for (_s, _m, c, e), n in grouped.items()
                                 if c != "UNCLEAR" and e in {"RULER_INCLUDED", "TOO_LARGE"})
        unassessable = sum(n for (_s, _m, _c, e), n in grouped.items() if e == "UNASSESSABLE")
        for model in ("v5", "v7"):
            overall = Counter()
            for row in rows:
                if row["model"] == model:
                    overall[(row["crop_condition"], row["error_type"])] += row["count"]
            rows.extend({"source_id": "ALL", "model": model, "crop_condition": c,
                         "error_type": e, "count": n} for (c, e), n in sorted(overall.items()))
        finished = complete == len(self.inputs.cases)
        recommendation = "Pričekati dovršetak vizualnog pregleda; još nema konačnog zaključka."
        if finished:
            if crop_counts["UNCLEAR"] == len(self.inputs.cases) or (
                not crop_counts["TRUNCATED"] and not complete_body_errors and not excess_mask_errors
            ):
                recommendation = "Najprije razjasniti vidljivost tijela u širem kontekstu."
            elif crop_counts["TRUNCATED"] and (complete_body_errors or excess_mask_errors):
                recommendation = ("Kombinacija: prvo probno proširiti crop/padding na odrezanim tijelima; "
                                  "za pogreške na cijelim tijelima pripremiti ciljane ručne maske. "
                                  "Fine-tuniranje razmotriti tek nakon provjere oznaka i potvrde korisnika.")
            elif crop_counts["TRUNCATED"]:
                recommendation = ("Najmanji prvi pokus: crop/padding na odrezanim tijelima, uz provjeru "
                                  "rezidualnih maski. Sam pregled ne dokazuje da padding rješava sve pogreške.")
            else:
                recommendation = ("Najprije dodatne ručne segmentacijske oznake stvarnih ruler pogrešaka; "
                                  "ciljano fine-tuniranje tek nakon njihove provjere i potvrde korisnika.")
            if crop_counts["UNCLEAR"] or unassessable:
                recommendation += " Nejasne slučajeve treba zasebno razjasniti prije promjene dataseta/modela."
        return {"scope": "57 previously rated R/BOTH_BAD cases, not an unbiased overall model comparison",
                "input_fingerprint": self.inputs.fingerprint,
                "completed_cases": complete, "total_cases": len(self.inputs.cases),
                "finished": finished, "crop_condition_counts": dict(crop_counts),
                "evidence": {"complete_body_model_errors": complete_body_errors,
                             "ruler_or_excess_mask_errors": excess_mask_errors,
                             "unassessable_model_ratings": unassessable},
                "counts_by_source_model_crop_and_error": rows, "next_step": recommendation,
                "training_ruler_scenes_are_independent_test": False}

    def export_summary(self):
        with self.lock:
            summary = self.summary()
            rows = summary["counts_by_source_model_crop_and_error"]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=["source_id", "model", "crop_condition", "error_type", "count"])
            writer.writeheader()
            writer.writerows(rows)
            for name, text in (("summary.json", json.dumps(summary, indent=2, ensure_ascii=False)),
                               ("summary_by_source_model_error.csv", buffer.getvalue())):
                temp = self.output / (name + ".tmp")
                temp.write_text(text, encoding="utf-8")
                os.replace(temp, self.output / name)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, inputs, store):
        self.inputs, self.store = inputs, store
        self.context_lock = threading.Lock()
        super().__init__(address, ReviewHandler)

    @lru_cache(maxsize=8)
    def context_image(self, index, wide):
        row = self.inputs.cases[index]
        box = self.inputs.context_box(row, wide)["box"]
        path = self.inputs.asset(self.inputs.sources[row["source_id"]]["image_path"])
        # Full-resolution JPEG decoding is serialized to bound peak RAM usage.
        with self.context_lock, Image.open(path) as image:
            crop = image.crop(box).convert("RGB")
        crop.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        crop.save(output, format="JPEG", quality=94)
        return output.getvalue()


class ReviewHandler(BaseHTTPRequestHandler):
    def send_body(self, body, content_type="application/json; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser navigation can cancel a large context response.

    def json_body(self, data, status=200):
        self.send_body(json.dumps(data, ensure_ascii=False).encode("utf-8"), status=status)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_body(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.json_body({"cases": self.server.inputs.public_cases(), "errors": ERRORS,
                                "crop_states": CROP_STATES, "audit": self.server.inputs.audit,
                                "state": self.server.store.state()})
            elif parsed.path == "/api/summary":
                self.json_body(self.server.store.summary())
            elif parsed.path == "/asset":
                query = parse_qs(parsed.query)
                index = int(query["index"][0])
                if not 0 <= index < len(self.server.inputs.cases):
                    raise ValueError("Invalid case")
                kind = query["kind"][0]
                row = self.server.inputs.cases[index]
                if kind == "context":
                    wide = query.get("wide", ["0"])[0] == "1"
                    self.send_body(self.server.context_image(index, wide), "image/jpeg")
                else:
                    column = {"original": "original_path", "v5": "v5_mask_path", "v7": "v7_mask_path"}.get(kind)
                    if column is None:
                        raise ValueError("Invalid asset kind")
                    if not row[column]:
                        self.json_body({"error": "No saved mask"}, 404)
                    else:
                        self.send_body(self.server.inputs.asset(row[column]).read_bytes(), "image/png")
            else:
                self.json_body({"error": "Not found"}, 404)
        except (ValueError, KeyError, IndexError) as exc:
            self.json_body({"error": str(exc)}, 400)
        except OSError as exc:
            self.json_body({"error": str(exc)}, 500)

    def do_POST(self):
        if self.path != "/api/decision":
            self.json_body({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get("Content-Type") != "application/json":
                raise ValueError("Expected small application/json request")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            self.json_body(self.server.store.record(data["case"], data["model"], data["category"]))
        except (ValueError, KeyError, TypeError) as exc:
            self.json_body({"error": str(exc)}, 400)
        except (OSError, sqlite3.Error) as exc:
            self.json_body({"error": str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/ruler_v7")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ruler_v7_body_error_review")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true", help="Read-only input validation, no GUI/server/output writes")
    args = parser.parse_args()
    inputs = ReviewInputs(args.reports_dir, args.assets_root)
    if args.check:
        print(json.dumps(inputs.audit, indent=2))
        return
    store = ReviewStore(inputs, args.output_dir)
    server = ReviewServer(("127.0.0.1", args.port), inputs, store)
    print(f"Ruler review: http://127.0.0.1:{server.server_port} (forward this port in VS Code)", flush=True)
    print(f"New decisions only: {store.database}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
