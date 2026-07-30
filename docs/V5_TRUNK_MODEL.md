# V5 Trunk Contour Model

This document describes how to run the Collembola trunk contour pipeline with the v5 segmentation model.

## Overview

The pipeline runs in two stages:

1. Tiled YOLO detector finds Collembola detections on full-resolution microscope images.
2. The v5 YOLO segmentation model predicts a trunk/body mask and contour for each detection crop.

The segmentation model is selected with the `--seg-model` CLI argument.

## Models

Detector model:

```text
models/yolo11n_tiled_best.pt
```

V5 trunk segmentation model:

```text
runs/segment/runs/segment/trunk_seg_v5_test20/weights/best.pt
```

## Python Environment

Install the required Python packages from the repository root:

```powershell
pip install -r requirements.txt
```

Main runtime dependencies include:

- `ultralytics`
- `torch`
- `torchvision`
- `opencv-python`
- `Pillow`
- `numpy`
- `pandas`

## Basic Run

Run raw v5 trunk segmentation on a folder of source images:

```powershell
python scripts\test_trunk_seg_generalization.py --images-dir "path\to\images" --detector-model "models\yolo11n_tiled_best.pt" --seg-model "runs\segment\runs\segment\trunk_seg_v5_test20\weights\best.pt" --output-dir "outputs\v5_trunk_test" --conf 0.6 --device cpu --imgsz 320
```

Use `--device 0` instead of `--device cpu` to run on GPU when CUDA is available.

## Experimental Envelope Variant

The script also includes an experimental envelope post-processing mode. This is still under development and should be treated as a review aid, not the default model output.

```powershell
python scripts\test_trunk_seg_generalization.py --images-dir "path\to\images" --detector-model "models\yolo11n_tiled_best.pt" --seg-model "runs\segment\runs\segment\trunk_seg_v5_test20\weights\best.pt" --output-dir "outputs\v5_trunk_test_envelope" --conf 0.6 --device cpu --imgsz 320 --use-envelope-contour
```

## Outputs

For each input source image, the script generates an output subfolder containing:

- tiled detector detections CSV and detector overlay under `detections/`
- per-detection crop images under `crops/`
- per-detection predicted crop overlays under `predicted_crops/`
- per-detection binary masks under `masks/`
- full-source trunk contour overlay image
- per-image segmentation results CSV
- contact sheet for quick visual review
- per-image `summary.json`

The root output folder also contains a combined `summary.json` across all processed images.

When `--use-envelope-contour` is enabled, additional envelope masks, envelope overlays, and comparison images are written to the envelope-specific output folders.

## Known Limitations

- Tile-boundary detections can be split or duplicated when a Collembola lies near a tile edge.
- The detector can produce false positives; those detections are still passed to the segmentation model.
- Cut, narrow, or poorly framed detection crops can produce poor trunk contours.
- The envelope mode is experimental post-processing and is not the default v5 model output.
