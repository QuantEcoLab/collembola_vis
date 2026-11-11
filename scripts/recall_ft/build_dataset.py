#!/usr/bin/env python3
"""Build the tiled YOLO fine-tuning datasets of the Pendifin detectors (v1 and v2).

Images: the TRAIN split of the Pendifin Čačinci (CAC) and Erdut (ERD) projects (replicates 1-3), with
CAC_PDF_Ctrl_2_002 left out (user-flagged evaluation image). Replicates 4-5 (holdout) are never read.

Labels per image:
  * annotation boxes with status accepted/added (rejected boxes are not labelled, i.e. negatives);
  * CAC images: plus the visually verified missed animals of verified_extras_v1.json ("animal");
    its "uncertain" objects are inpainted, so they are neither positives nor negatives;
  * a box that lies >= 80% inside a larger box of the same image is dropped (fragment duplicate).
Tiles: 1280 px with stride 1024 (as at inference). A box is labelled in a tile when >= 40% of its area
is inside (clipped); smaller visible parts are inpainted. Every labelled tile is kept plus a seeded 35%
of unlabelled tiles (substrate texture, fibres, ruler, grid paper, dish rim = hard negatives). For a
seeded 30% of labelled tiles a 640 crop around a random label is upscaled 2x, so the detector also
works on the 640-tile pass. Tiles overlapping excluded_dev_tiles.json are dropped.

v2 (models/yolo11n_tiled_pendifin_v2.pt) adds --verified2 verified_r2.json (round-2 visual checks of
the v1 detections on the same train images):
  * its "animal" objects become labels, its "uncertain" objects are inpainted (unless they overlap a
    labelled animal by more than 10% of the smaller box);
  * "drop_pos" removes a positive label (IoU >= 0.9): shapeless colour-fringed blobs and boxes on the grid
    card that were labelled as animals;
  * for every "not" object (texture, dish rim/halo, fibre, grid, colour-fringed blob, partial box on an
    animal) one extra 1280 tile is cut with the object at a seeded random position (hard negative).
These extra tiles use their own random generator, so without --verified2 the v1 set is rebuilt exactly.

Usage:
  python scripts/recall_ft/build_dataset.py --split-file split.json --annotations-dir data/annotations \
      --uploads-dir data/uploads --out data/recall_ft_v1 [--orig data/yolo_tiled]
  v2: add --verified2 scripts/recall_ft/verified_r2.json
--orig adds the original training set (train + val dirs) to data.yaml so the model keeps working on
the plates it was first trained on.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
T, S = 1280, 1024
EXCLUDED_STEMS = {"CAC_PDF_Ctrl_2_002"}


def defrag(b):
    """Drop boxes lying >= 80% inside a larger box."""
    if len(b) == 0:
        return b
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = np.ones(len(b), bool)
    for i in range(len(b)):
        inter = (np.clip(np.minimum(b[:, 2], b[i, 2]) - np.maximum(b[:, 0], b[i, 0]), 0, None)
                 * np.clip(np.minimum(b[:, 3], b[i, 3]) - np.maximum(b[:, 1], b[i, 1]), 0, None))
        frac = inter / area[i]
        frac[i] = 0
        if ((frac >= 0.8) & (area > area[i])).any():
            keep[i] = False
    return b[keep]


def grid(W, H):
    xs = list(range(0, max(W - T, 0) + 1, S))
    if xs[-1] + T < W:
        xs.append(W - T)
    ys = list(range(0, max(H - T, 0) + 1, S))
    if ys[-1] + T < H:
        ys.append(H - T)
    return [(x, y) for y in ys for x in xs]


def overlaps(x0, y0, w, h, rects):
    return any(x0 < r[0] + r[2] and r[0] < x0 + w and y0 < r[1] + r[2] and r[1] < y0 + h for r in rects)


def labels_in(pos, ign, x0, y0, w, h):
    lab, extra_ign = [], []
    for b in pos:
        cb = [max(b[0], x0), max(b[1], y0), min(b[2], x0 + w), min(b[3], y0 + h)]
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            continue
        frac = (cb[2] - cb[0]) * (cb[3] - cb[1]) / ((b[2] - b[0]) * (b[3] - b[1]))
        (lab if frac >= 0.4 else extra_ign).append(cb)
    ig = [b for b in ign if b[2] > x0 and b[0] < x0 + w and b[3] > y0 and b[1] < y0 + h] + extra_ign
    return lab, ig


def inpaint(tile, ig, x0, y0):
    if not ig:
        return tile
    m = np.zeros(tile.shape[:2], np.uint8)
    for b in ig:
        cv2.rectangle(m, (int(b[0] - x0 - 4), int(b[1] - y0 - 4)), (int(b[2] - x0 + 4), int(b[3] - y0 + 4)), 255, -1)
    return cv2.inpaint(tile, m, 7, cv2.INPAINT_TELEA)


def write(out, name, img, lab, x0, y0, scale):
    cv2.imwrite(f"{out}/images/train/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(f"{out}/labels/train/{name}.txt", "w") as f:
        for b in lab:
            cx, cy = ((b[0] + b[2]) / 2 - x0) * scale / T, ((b[1] + b[3]) / 2 - y0) * scale / T
            w, h = (b[2] - b[0]) * scale / T, (b[3] - b[1]) * scale / T
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def iou_one(b, B):
    B = np.asarray(B, float).reshape(-1, 4)
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(B[:, 2], b[2]) - np.maximum(B[:, 0], b[0]), 0, None)
    iy = np.clip(np.minimum(B[:, 3], b[3]) - np.maximum(B[:, 1], b[1]), 0, None)
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


def overlap_small(b, B):
    """Largest intersection of b with a box of B, as a fraction of the smaller of the two boxes."""
    B = np.asarray(B, float).reshape(-1, 4)
    if len(B) == 0:
        return 0.0
    ix = np.clip(np.minimum(B[:, 2], b[2]) - np.maximum(B[:, 0], b[0]), 0, None)
    iy = np.clip(np.minimum(B[:, 3], b[3]) - np.maximum(B[:, 1], b[1]), 0, None)
    small = np.minimum((b[2] - b[0]) * (b[3] - b[1]), (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]))
    return float((ix * iy / small).max())


def hard_negative_tiles(out, stem, img, pos, ign, negs, seed, excl):
    """One 1280 tile per hard-negative object, the object at a seeded random position inside it."""
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    n = 0
    for k, b in enumerate(negs):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(np.clip(cx - rng.integers(160, T - 160), 0, W - T))
        y0 = int(np.clip(cy - rng.integers(160, T - 160), 0, H - T))
        if overlaps(x0, y0, T, T, excl):
            continue
        lab, ig = labels_in(pos, ign, x0, y0, T, T)
        write(out, f"{stem}_hn{k}_{x0}_{y0}", inpaint(np.ascontiguousarray(img[y0:y0 + T, x0:x0 + T]), ig, x0, y0),
              lab, x0, y0, 1.0)
        n += 1
    return n


def process(job):
    stem, path, pos, ign, out, seed, excl, negs = job
    rng = np.random.default_rng(seed)
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    H, W = img.shape[:2]
    n = [0, 0, 0]
    for (x0, y0) in grid(W, H):
        if overlaps(x0, y0, T, T, excl):
            continue
        lab, ig = labels_in(pos, ign, x0, y0, T, T)
        if not lab and rng.random() > 0.35:
            continue
        write(out, f"{stem}_{x0}_{y0}", inpaint(np.ascontiguousarray(img[y0:y0 + T, x0:x0 + T]), ig, x0, y0), lab, x0, y0, 1.0)
        n[0 if lab else 1] += 1
        if lab and rng.random() < 0.3:
            b = lab[rng.integers(len(lab))]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ux = int(np.clip(cx - rng.integers(80, 560), 0, W - 640))
            uy = int(np.clip(cy - rng.integers(80, 560), 0, H - 640))
            if overlaps(ux, uy, 640, 640, excl):
                continue
            lab2, ig2 = labels_in(pos, ign, ux, uy, 640, 640)
            crop = cv2.resize(inpaint(np.ascontiguousarray(img[uy:uy + 640, ux:ux + 640]), ig2, ux, uy), (T, T),
                              interpolation=cv2.INTER_LINEAR)
            write(out, f"{stem}_{x0}_{y0}_up{ux}_{uy}", crop, lab2, ux, uy, 2.0)
            n[2] += 1
    n.append(hard_negative_tiles(out, stem, img, pos, ign, negs, seed + 50000, excl) if negs else 0)
    return stem, len(pos), len(ign), *n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", required=True, help="critic split.json (image list with split/replicate)")
    ap.add_argument("--annotations-dir", required=True, help="dir with <image_id>.json annotation files")
    ap.add_argument("--uploads-dir", required=True)
    ap.add_argument("--verified", default=str(HERE / "verified_extras_v1.json"))
    ap.add_argument("--verified2", default=None, help="round-2 verified objects (v2), e.g. verified_r2.json")
    ap.add_argument("--exclude-tiles", default=str(HERE / "excluded_dev_tiles.json"))
    ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--orig", default=None, help="original YOLO dataset dir (images/train, images/val)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    reps = {int(x) for x in a.reps.split(",")}
    verified = json.load(open(a.verified))["images"]
    verified2 = json.load(open(a.verified2))["images"] if a.verified2 else {}
    excl = {}
    for t in json.load(open(a.exclude_tiles))["tiles"]:
        excl.setdefault(t["stem"], []).append((t["x0"], t["y0"], t["size"]))
    for sub in ("images/train", "labels/train"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    jobs = []
    train = [e for e in json.load(open(a.split_file))["images"] if e["split"] == "train" and e["stem"] not in EXCLUDED_STEMS]
    for i, e in enumerate(train):  # seed = 1000 + position in this list (as used for the v1 dataset)
        if e["replicate"] not in reps:
            continue
        d = json.load(open(Path(a.annotations_dir) / f"{e['image_id']}.json"))
        pos = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in d["boxes"] if b["status"] != "rejected"]
        v = verified.get(e["stem"], {"animal": [], "uncertain": []})
        pos = pos + [b[:4] for b in v["animal"]]
        ign = [b[:4] for b in v["uncertain"]]
        negs = []
        if a.verified2:
            v2 = verified2.get(e["stem"], {})
            for b in v2.get("drop_pos", []):
                pos = [p for p in pos if not iou_one(p, [b[:4]])[0] >= 0.9]
            pos += [b[:4] for b in v2.get("animal", [])]
            # An uncertain object touching a labelled animal is not inpainted (that would erase the animal).
            ign += [b[:4] for b in v2.get("uncertain", []) if overlap_small(b[:4], pos) <= 0.1]
            negs = [b[:4] for b in v2.get("not", [])]
        pos = defrag(np.array(pos, float).reshape(-1, 4)).tolist()
        jobs.append((e["stem"], str(Path(a.uploads_dir) / e["image_id"] / e["filename"]), pos, ign, a.out, 1000 + i,
                     excl.get(e["stem"], []), negs))
    with Pool(a.workers) as p:
        for r in p.imap_unordered(process, jobs):
            print("{}\tlabels={} inpainted={} tiles: labelled={} empty={} upscaled={} hard_neg={}".format(*r), flush=True)
    lines = [f"path: {a.out}", "train:", f"  - {a.out}/images/train"]
    if a.orig:
        lines += [f"  - {a.orig}/images/train", "val:", f"  - {a.orig}/images/val"]
    else:
        lines += ["val:", f"  - {a.out}/images/train"]
    open(f"{a.out}/data.yaml", "w").write("\n".join(lines + ["nc: 1", "names: ['collembola']"]) + "\n")


if __name__ == "__main__":
    main()
