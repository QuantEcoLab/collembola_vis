#!/usr/bin/env python3
"""Run the ruler crop rectangle selector for all active LabelMe annotations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolved(path: Path) -> Path:
    return path.resolve()


def load_active_labelme(crop_metadata: Path, labelme_dir: Path) -> list[Path]:
    rows = list(csv.DictReader(crop_metadata.open(newline="", encoding="utf-8")))
    out: list[Path] = []
    missing: list[str] = []
    for row in rows:
        stem = Path(row["crop_file"].replace("\\", "/")).stem
        json_path = labelme_dir / f"{stem}.json"
        if json_path.exists():
            out.append(json_path)
        else:
            missing.append(str(json_path))
    if missing:
        print("Missing LabelMe JSON files:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(1)
    return out


def completed_sources(output_dir: Path) -> set[Path]:
    done: set[Path] = set()
    if not output_dir.exists():
        return done
    for meta_path in output_dir.glob("selection_*/selection_metadata.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = data.get("source_labelme_json")
        if source:
            normalized = source.replace("\\", "/")
            path = Path(normalized)
            if not path.is_absolute() and len(normalized) > 1 and normalized[1] == ":":
                # Restore the repo-relative suffix of historical Windows records.
                _prefix, separator, suffix = normalized.partition("/data/")
                if not separator:
                    raise ValueError(f"Cannot resolve saved source path: {source}")
                path = ROOT / "data" / suffix
            elif not path.is_absolute():
                path = ROOT / path
            done.add(path.resolve())
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select training crop rectangles for all active ruler LabelMe annotations")
    parser.add_argument("--crop-metadata", type=Path, default=ROOT / "reports/ruler_v7/ruler/crop_metadata.csv")
    parser.add_argument("--labelme-dir", type=Path, default=ROOT / "data/ruler_manual_annotation_v1/labelme")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/ruler_manual_annotation_v1/review/manual_rect_selection")
    parser.add_argument("--redo-completed", action="store_true", help="Open images even if a prior selection already exists.")
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=850)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labelme_paths = load_active_labelme(args.crop_metadata, args.labelme_dir)
    done = completed_sources(args.output_dir)
    selector = Path(__file__).with_name("select_ruler_crop_review.py")

    pending = []
    for path in labelme_paths:
        if not args.redo_completed and path.resolve() in done:
            print(f"Skipping completed: {path}")
            continue
        pending.append(path)

    if not pending:
        print("All active crops already have at least one saved selection. Use --redo-completed to review again.")
        return

    print(f"Opening {len(pending)} crop(s) for rectangle selection.")
    for idx, path in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] {path}")
        cmd = [
            sys.executable,
            str(selector),
            "--labelme-json",
            str(path),
            "--output-dir",
            str(args.output_dir),
            "--max-width",
            str(args.max_width),
            "--max-height",
            str(args.max_height),
            "--exit-after-save",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Stopped before finishing all crops.")
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
