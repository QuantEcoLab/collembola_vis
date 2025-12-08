from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image


def find_plate_for_prefix(plates_dir: Path, prefix: str) -> Path | None:
    candidates = [p for p in plates_dir.glob("*.jpg") if p.name.startswith(prefix)]
    if candidates:
        return candidates[0]
    for p in plates_dir.glob("*.jpg"):
        if prefix in p.stem:
            return p
    return None


def main():
    ann_csv = Path("data/annotations/collembolas_table.csv")
    plates_dir = Path("data/slike")
    out_dir = Path("data/crops/gt_positive")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ann_csv)
    if not set(["id_collembole", "x", "y", "w", "h"]).issubset(df.columns):
        raise SystemExit("collembolas_table.csv must have id_collembole,x,y,w,h")

    df["prefix"] = df["id_collembole"].astype(str).apply(lambda s: s.split("_c_")[0])
    groups = df.groupby("prefix")

    saved = 0
    missing = []

    Image.MAX_IMAGE_PIXELS = None

    for prefix, g in groups:
        plate_path = find_plate_for_prefix(plates_dir, prefix)
        if plate_path is None:
            missing.append(prefix)
            continue
        im = Image.open(plate_path).convert("RGB")
        W, H = im.size
        for _, r in g.iterrows():
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = im.crop((x0, y0, x1, y1))
            out_path = out_dir / f"{r['id_collembole']}.jpg"
            crop.save(out_path)
            saved += 1

    print({"saved": saved, "missing_prefixes": sorted(set(missing))})


if __name__ == "__main__":
    main()
