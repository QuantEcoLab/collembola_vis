#!/usr/bin/env python3
"""
Tiled inference for large collembola images.

This script:
1. Tiles large images with overlap
2. Runs YOLO detection on each tile
3. Merges predictions with NMS across tile boundaries
4. Outputs full-image detections in CSV and overlay visualization

Usage:
    python scripts/infer_tiled.py --image path/to/image.jpg --model runs/detect/train_tiled_1280_20251210_115016/weights/best.pt
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch


def ultralytics_device(device):
    """Return the ``device`` argument to hand to ultralytics.

    ``select_device("cpu")`` sets ``os.environ["CUDA_VISIBLE_DEVICES"] = ""`` for
    the whole process. Inside the long-running web backend that hides every GPU
    from later jobs (torch caches the empty device list), so CPU runs pass a
    ``torch.device``, which ultralytics uses as is. The CPU thread count is set
    exactly as ``select_device("cpu")`` would set it.
    """
    if isinstance(device, str) and device.strip().lower() == "cpu":
        from ultralytics.utils import NUM_THREADS

        torch.set_num_threads(NUM_THREADS)
        return torch.device("cpu")
    return device


def tile_positions(img_w, img_h, tile_size=1280, overlap=256):
    """Return the (x_start, y_start, x_end, y_end) grid used by ``tile_image``.

    Tiles advance by ``tile_size - overlap``; the last tile in each row/column is
    shifted back so that it ends exactly at the image border.
    """
    positions = []
    stride = tile_size - overlap
    y = 0
    while y < img_h:
        x = 0
        y_end = min(y + tile_size, img_h)
        y_start = max(0, y_end - tile_size) if y_end == img_h else y
        while x < img_w:
            x_end = min(x + tile_size, img_w)
            x_start = max(0, x_end - tile_size) if x_end == img_w else x
            positions.append((x_start, y_start, x_end, y_end))
            if x_end == img_w:
                break
            x += stride
        if y_end == img_h:
            break
        y += stride
    return positions


def tile_image(image_path, tile_size=1280, overlap=256):
    """
    Tile a large image into smaller overlapping tiles.

    Returns:
        img: PIL Image
        tiles: List of (tile_img, x_offset, y_offset, tile_id)
    """
    img = Image.open(image_path)
    tiles = []
    for tile_id, (x0, y0, x1, y1) in enumerate(tile_positions(*img.size, tile_size, overlap)):
        tiles.append((img.crop((x0, y0, x1, y1)), x0, y0, tile_id))
    return img, tiles


def nms_global(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression across all tiles.
    
    Args:
        detections: List of dicts with keys: x1, y1, x2, y2, conf, class
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of kept detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections])
    scores = np.array([d['conf'] for d in detections])
    
    # Sort by confidence
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Pick detection with highest confidence
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_order - inter
        
        iou = inter / union
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [detections[i] for i in keep]


def suppress_contained(detections, contain_threshold=0.8):
    """Drop fragment boxes: a box is removed when at least ``contain_threshold`` of its
    area lies inside a larger box that is kept (largest boxes are considered first).

    Tiled and multi-scale inference often returns a partial box on one animal next to the
    full box (tile borders, body parts); NMS keeps both because their IoU is small.
    """
    if not detections or contain_threshold is None or contain_threshold >= 1:
        return list(detections)
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections], dtype=float)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    for i in np.argsort(-areas, kind='stable'):
        if kept:
            k = boxes[kept]
            inter = (np.clip(np.minimum(k[:, 2], boxes[i, 2]) - np.maximum(k[:, 0], boxes[i, 0]), 0, None)
                     * np.clip(np.minimum(k[:, 3], boxes[i, 3]) - np.maximum(k[:, 1], boxes[i, 1]), 0, None))
            if areas[i] > 0 and (inter / areas[i]).max() >= contain_threshold:
                continue
        kept.append(i)
    return [detections[i] for i in sorted(kept)]


def detect_pass(model, img, tile_size, overlap, conf_threshold, model_device, pass_id=0, progress=None):
    """Run the detector over one tile grid of ``img``; boxes in full-image pixels."""
    detections = []
    positions = tile_positions(*img.size, tile_size, overlap)
    for tile_idx, (x0, y0, x1, y1) in enumerate(positions):
        if progress:
            progress(tile_idx, len(positions))
        results = model(img.crop((x0, y0, x1, y1)), conf=conf_threshold, device=model_device, verbose=False)
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'x1': float(bx1 + x0),
                    'y1': float(by1 + y0),
                    'x2': float(bx2 + x0),
                    'y2': float(by2 + y0),
                    'conf': float(boxes.conf[i].cpu().numpy()),
                    'class': int(boxes.cls[i].cpu().numpy()),
                    'tile_id': tile_idx,
                    'pass': pass_id,
                })
    return detections, len(positions)


def infer_tiled(
    image_path,
    model_path,
    tile_size=1280,
    overlap=256,
    conf_threshold=0.25,
    iou_threshold=0.5,
    output_dir=None,
    device='0',
    progress_callback=None,
    extra_tile_sizes=None,
    extra_conf_thresholds=None,
    contain_threshold=None,
):
    """
    Run tiled inference on a large image.

    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        tile_size: Size of tiles
        overlap: Overlap between tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        output_dir: Output directory (default: infer_tiled_output)
        device: GPU device(s) to use
        progress_callback: Optional callable(progress: float, message: str) for progress updates
        extra_tile_sizes: Optional further tile sizes run with the same model (e.g. [640]: the
            model's 1280 input then sees each 640 tile upscaled 2x, which finds small juveniles)
        extra_conf_thresholds: Confidence threshold for each extra tile size
        contain_threshold: If set, drop boxes that lie at least this fraction inside a larger box
            (fragment duplicates), applied after the global NMS

    Returns:
        detections: List of detection dictionaries
    """
    # Setup
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = Path('infer_tiled_output')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_name = image_path.stem

    print(f"Processing: {image_path}")
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading model: {model_path}")
    if progress_callback:
        progress_callback(0.0, "Loading model...")
    model = YOLO(model_path)
    model_device = ultralytics_device(device)

    # Tile sizes to run: the main grid plus optional extra scales
    extra_tile_sizes = list(extra_tile_sizes or [])
    extra_conf_thresholds = list(extra_conf_thresholds or [])
    if len(extra_conf_thresholds) != len(extra_tile_sizes):
        raise ValueError("extra_conf_thresholds needs one value per extra tile size")
    passes = [(int(tile_size), float(conf_threshold))] + [
        (int(t), float(c)) for t, c in zip(extra_tile_sizes, extra_conf_thresholds)]

    if progress_callback:
        progress_callback(0.04, "Tiling image...")
    img = Image.open(image_path)
    img.load()
    img_w, img_h = img.size
    pass_tiles = [len(tile_positions(img_w, img_h, t, overlap)) for t, _ in passes]
    total_tiles = sum(pass_tiles)
    n_tiles = pass_tiles[0]
    print(f"Image size: {img_w}x{img_h}")
    print(f"Tile passes (tile_size, conf): {passes}, overlap={overlap}, {total_tiles} tiles")

    # Run detection on each tile of each pass
    print(f"\nRunning detection on tiles...")
    if progress_callback:
        progress_callback(0.07, f"Running detection on {total_tiles} tiles...")
    all_detections = []
    done = 0
    for pass_id, (pass_tile, pass_conf) in enumerate(passes):
        def progress(tile_idx, n, _done=done):
            # Report progress (7% -> 87% across all tiles of all passes)
            if progress_callback:
                progress_callback(0.07 + 0.80 * ((_done + tile_idx) / total_tiles), f"Tile {_done + tile_idx + 1}/{total_tiles}")
        dets, n = detect_pass(model, img, pass_tile, overlap, pass_conf, model_device, pass_id, progress)
        all_detections.extend(dets)
        done += n

    print(f"Total detections before NMS: {len(all_detections)}")

    # Apply global NMS
    print(f"\nApplying global NMS (IoU threshold={iou_threshold})...")
    if progress_callback:
        progress_callback(0.88, f"Applying NMS to {len(all_detections)} candidates...")
    final_detections = nms_global(all_detections, iou_threshold)
    n_after_nms = len(final_detections)
    print(f"Final detections after NMS: {n_after_nms}")
    if contain_threshold is not None:
        final_detections = suppress_contained(final_detections, contain_threshold)
        print(f"After fragment suppression (contain >= {contain_threshold}): {len(final_detections)}")

    # Save to CSV
    csv_path = output_dir / f"{image_name}_detections.csv"
    df_data = []
    for det in final_detections:
        df_data.append({
            'x1': det['x1'],
            'y1': det['y1'],
            'x2': det['x2'],
            'y2': det['y2'],
            'width': det['x2'] - det['x1'],
            'height': det['y2'] - det['y1'],
            'confidence': det['conf'],
            'class': det['class']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detections to: {csv_path}")
    
    # Create visualization
    print(f"\nCreating visualization...")
    if progress_callback:
        progress_callback(0.93, f"Creating overlay ({len(final_detections)} detections)...")
    draw = ImageDraw.Draw(img)
    
    for det in final_detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw confidence score
        text = f"{conf:.2f}"
        draw.text((x1, y1 - 10), text, fill='red')
    
    # Save overlay
    if progress_callback:
        progress_callback(0.97, "Saving results...")
    overlay_path = output_dir / f"{image_name}_overlay.jpg"
    img.save(overlay_path, quality=95)
    print(f"Saved overlay to: {overlay_path}")

    # Save metadata
    metadata = {
        'image_path': str(image_path),
        'image_size': [img_w, img_h],
        'model_path': str(model_path),
        'tile_size': tile_size,
        'overlap': overlap,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'num_tiles': n_tiles,
        'extra_tile_sizes': [t for t, _ in passes[1:]],
        'extra_conf_thresholds': [c for _, c in passes[1:]],
        'contain_threshold': contain_threshold,
        'num_tiles_all_passes': total_tiles,
        'detections_before_nms': len(all_detections),
        'detections_after_nms': n_after_nms,
        'detections_final': len(final_detections)
    }
    
    metadata_path = output_dir / f"{image_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    if progress_callback:
        progress_callback(1.0, f"Done — {len(final_detections)} detections")

    print("\n" + "="*70)
    print(f"Inference complete!")
    print(f"Detected {len(final_detections)} collembola organisms")
    print("="*70)

    return final_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiled inference for large collembola images')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, 
                        default='runs/detect/train_tiled_1280_20251210_115016/weights/best.pt',
                        help='Path to YOLO model weights')
    parser.add_argument('--tile-size', type=int, default=1280,
                        help='Tile size (default: 1280)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between tiles (default: 256)')
    parser.add_argument('--conf', type=float, default=0.6,
                        help='Confidence threshold (default: 0.6)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--output', type=str, default='infer_tiled_output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device to use (default: 0)')
    parser.add_argument('--extra-tile-sizes', type=int, nargs='*', default=[],
                        help='Extra tile sizes run with the same model, e.g. 640 (default: none)')
    parser.add_argument('--extra-conf', type=float, nargs='*', default=[],
                        help='Confidence threshold for each extra tile size')
    parser.add_argument('--contain', type=float, default=None,
                        help='Drop boxes lying at least this fraction inside a larger box, e.g. 0.8 (default: off)')

    args = parser.parse_args()
    
    infer_tiled(
        image_path=args.image,
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
        device=args.device,
        extra_tile_sizes=args.extra_tile_sizes,
        extra_conf_thresholds=args.extra_conf,
        contain_threshold=args.contain,
    )
