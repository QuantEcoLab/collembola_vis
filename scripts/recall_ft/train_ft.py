#!/usr/bin/env python3
"""Fine-tune the production detector on the dataset of build_dataset.py (single GPU).

v1 (models/yolo11n_tiled_pendifin_v1.pt) was trained with:
  python scripts/recall_ft/train_ft.py --data <out>/data.yaml --epochs 25 --batch 4 --device 0
from models/yolo11n_tiled_best.pt on one Quadro RTX 8000 (about 45 min); the weights are last.pt.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="models/yolo11n_tiled_best.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.003)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/recall_ft")
    ap.add_argument("--name", default="pendifin_v1")
    a = ap.parse_args()
    YOLO(a.base).train(
        data=a.data, epochs=a.epochs, imgsz=1280, batch=a.batch, device=a.device, workers=10,
        project=a.project, name=a.name, exist_ok=True, pretrained=True, optimizer="SGD",
        lr0=a.lr0, lrf=0.1, momentum=0.937, weight_decay=5e-4, warmup_epochs=1.0, cos_lr=True,
        close_mosaic=5, single_cls=True, fliplr=0.5, flipud=0.5, mosaic=1.0, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=0.0, mixup=0.0, copy_paste=0.0, amp=True,
        cache="ram", seed=42, deterministic=False, plots=False, val=True, patience=100, save_period=5)


if __name__ == "__main__":
    main()
