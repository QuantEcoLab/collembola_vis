#!/usr/bin/env python3
"""Fine-tune the production detector on a dataset of build_dataset.py (single GPU).

Pick the GPU with CUDA_VISIBLE_DEVICES; --device 0 then means the first *visible* GPU:
  CUDA_VISIBLE_DEVICES=2 python scripts/recall_ft/train_ft.py --data <out>/data.yaml --name pendifin_v2

Ultralytics' select_device("0") writes CUDA_VISIBLE_DEVICES="0" into the environment before torch
initialises CUDA, which would silently move the job to physical GPU 0 (on the production server that
is the GPU the web service uses). This script therefore requires CUDA_VISIBLE_DEVICES to be set,
initialises CUDA first so the visible GPU list is fixed for the process, and prints the physical
GPU it trains on.

v1 (models/yolo11n_tiled_pendifin_v1.pt) and v2 (models/yolo11n_tiled_pendifin_v2.pt) were trained
from models/yolo11n_tiled_best.pt with the defaults below (25 epochs, batch 4, imgsz 1280; about
45 min on one Quadro RTX 8000); the weights are last.pt.
"""
import argparse
import os
import sys


def pin_visible_gpu(device: str) -> str:
    """Fix the visible GPU list before ultralytics can rewrite CUDA_VISIBLE_DEVICES."""
    if device.strip().lower() == "cpu":
        return "cpu"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        sys.exit("Set CUDA_VISIBLE_DEVICES to the GPU to train on (e.g. CUDA_VISIBLE_DEVICES=2); "
                 "without it --device 0 is physical GPU 0.")
    import torch

    count = torch.cuda.device_count()  # first CUDA query: the visible set is now fixed for this process
    index = int(device)
    if index >= count:
        sys.exit(f"--device {device} is not visible (CUDA_VISIBLE_DEVICES={visible!r}, {count} GPU(s))")
    torch.zeros(1, device=f"cuda:{index}")
    props = torch.cuda.get_device_properties(index)
    print(f"[train_ft] CUDA_VISIBLE_DEVICES={visible} -> cuda:{index} = {props.name} uuid={props.uuid}", flush=True)
    return str(index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="models/yolo11n_tiled_best.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.003)
    ap.add_argument("--device", default="0", help="index among the visible GPUs (see CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--project", default="runs/recall_ft")
    ap.add_argument("--name", default="pendifin_v1")
    a = ap.parse_args()
    device = pin_visible_gpu(a.device)

    from ultralytics import YOLO

    YOLO(a.base).train(
        data=a.data, epochs=a.epochs, imgsz=1280, batch=a.batch, device=device, workers=10,
        project=a.project, name=a.name, exist_ok=True, pretrained=True, optimizer="SGD",
        lr0=a.lr0, lrf=0.1, momentum=0.937, weight_decay=5e-4, warmup_epochs=1.0, cos_lr=True,
        close_mosaic=5, single_cls=True, fliplr=0.5, flipud=0.5, mosaic=1.0, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=0.0, mixup=0.0, copy_paste=0.0, amp=True,
        cache="ram", seed=42, deterministic=False, plots=False, val=True, patience=100, save_period=5)


if __name__ == "__main__":
    main()
