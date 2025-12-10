from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

from collembola_pipeline.config import CSV_DIR, ANNOTATIONS_DIR


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(dets, gts, thr=0.5):
    matched = set()
    TP = 0
    FP = 0
    FN = 0
    for i, d in enumerate(dets):
        best = -1
        best_iou = 0
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = iou(d, g)
            if v > best_iou:
                best_iou = v
                best = j
        if best_iou >= thr:
            matched.add(best)
            TP += 1
        else:
            FP += 1
    FN = len(gts) - TP
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--plate', type=str, required=True, help='Plate stem name (without extension)')
    p.add_argument('--iou', type=float, default=0.5)
    args = p.parse_args()

    det_csv = CSV_DIR / f"{args.plate}_organisms.csv"
    df_det = pd.read_csv(det_csv)
    dets = df_det[['bbox_x','bbox_y','bbox_w','bbox_h']].values.tolist()

    gt_csv = ANNOTATIONS_DIR / 'collembolas_table.csv'
    df_gt = pd.read_csv(gt_csv)
    # Filter ground truth to the given plate
    m = df_gt.apply(lambda r: args.plate in str(r).lower(), axis=1)
    df_gt = df_gt[m]
    # Try to find bbox columns; fallback if missing
    cols = [c for c in df_gt.columns]
    xc = next((c for c in cols if c.lower() in ('x','bbox_x','x1')), None)
    yc = next((c for c in cols if c.lower() in ('y','bbox_y','y1')), None)
    wc = next((c for c in cols if 'w' in c.lower()), None)
    hc = next((c for c in cols if 'h' in c.lower()), None)
    if None in (xc, yc, wc, hc):
        print('Could not locate bbox columns in ground truth CSV')
        exit(1)
    gts = df_gt[[xc,yc,wc,hc]].astype(int).values.tolist()

    metrics = match(dets, gts, thr=args.iou)
    print(json.dumps(metrics, indent=2))
