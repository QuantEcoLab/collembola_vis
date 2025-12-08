from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from skimage.transform import resize

from .config import (
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    SAM_AUTOMASK_PARAMS,
    SAM_AUTOMASK_GPU_PARAMS,
    SAM_AUTOMASK_CPU_PARAMS,
    DOWNSCALE_MAX_SIDE,
)


def _maybe_downscale(
    image: np.ndarray, device: str, max_side: int
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if device.startswith("cpu") or device == "cpu":
        ms = max(h, w)
        if ms > max_side:
            scale = max_side / float(ms)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            # skimage resize returns float; preserve range then cast back
            img_ds = resize(
                image,
                (new_h, new_w),
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(image.dtype)
            return img_ds, scale
    return image, scale


def _rescale_masks_to_original(
    masks: List[Dict[str, Any]], original_hw: Tuple[int, int], scale: float
) -> List[Dict[str, Any]]:
    if scale == 1.0:
        return masks
    oh, ow = original_hw
    inv = 1.0 / scale
    out: List[Dict[str, Any]] = []
    for m in masks:
        seg = m.get("segmentation")
        bbox = m.get("bbox", [0, 0, 0, 0])
        if seg is not None:
            seg_up = resize(
                seg.astype(float),
                (oh, ow),
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ).astype(bool)
        else:
            seg_up = None
        x, y, w, h = bbox
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))
        mm = dict(m)
        mm["bbox"] = [x, y, w, h]
        if seg_up is not None:
            mm["segmentation"] = seg_up
            mm["area"] = int(seg_up.sum())
        out.append(mm)
    return out


def load_sam(device: str = "cuda"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    # Choose params based on device to aim for higher recall on GPU
    if device.startswith("cuda") or device == "cuda":
        params = dict(SAM_AUTOMASK_GPU_PARAMS)
    else:
        params = dict(SAM_AUTOMASK_CPU_PARAMS)
    # Fallback to base params where override missing
    base = dict(SAM_AUTOMASK_PARAMS)
    base.update(params)
    mask_generator = SamAutomaticMaskGenerator(model=sam, **base)
    return mask_generator


def generate_masks(
    image: np.ndarray,
    device: str = "cuda",
    auto_downscale_cpu: bool = True,
    downscale_max_side: int = DOWNSCALE_MAX_SIDE,
) -> List[Dict[str, Any]]:
    """
    Generate SAM masks for an image.
    - CPU: optionally downscale large images to avoid OOM.
    - GPU: for very large images, process in overlapping tiles to avoid CUDA OOM.
    - Masks' bboxes are mapped to original coordinates; segmentations remain per-mask arrays.
    """
    from .config import TILE_MAX_SIDE, TILE_OVERLAP

    def iou(b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = max(0, w1) * max(0, h1)
        a2 = max(0, w2) * max(0, h2)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def dedup_by_iou(
        masks_list: List[Dict[str, Any]], thr: float = 0.7
    ) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for m in masks_list:
            b = m.get("bbox", [0, 0, 0, 0])
            area = int(m.get("area", 0))
            if area == 0 and m.get("segmentation") is not None:
                area = int(np.asarray(m["segmentation"]).sum())
                m["area"] = area
            drop = False
            for i, km in enumerate(kept):
                bi = km.get("bbox", [0, 0, 0, 0])
                if iou(b, bi) >= thr:
                    # keep the larger area
                    ka = int(km.get("area", 0))
                    if ka == 0 and km.get("segmentation") is not None:
                        ka = int(np.asarray(km["segmentation"]).sum())
                        km["area"] = ka
                    if area > ka:
                        kept[i] = m
                    drop = True
                    break
            if not drop:
                kept.append(m)
        return kept

    H, W = image.shape[:2]

    # CPU path: optional downscale to avoid OOM
    if device == "cpu" or device.startswith("cpu"):
        original_hw = (H, W)
        img_in, scale = (image, 1.0)
        if auto_downscale_cpu:
            img_in, scale = _maybe_downscale(image, device, downscale_max_side)
        mask_generator = load_sam(device=device)
        masks = mask_generator.generate(img_in)
        masks = _rescale_masks_to_original(masks, original_hw, scale)
        return masks

    # GPU path: tile if image is very large
    max_side = max(H, W)
    mask_generator = load_sam(device=device)
    if max_side <= TILE_MAX_SIDE:
        return mask_generator.generate(image)

    # Build tile start indices with overlap and coverage to the end
    def starts(total: int, size: int, overlap: int):
        stride = max(1, size - overlap)
        s = list(range(0, max(1, total - size + 1), stride))
        if not s or s[-1] + size < total:
            s.append(max(0, total - size))
        return s

    tiles_y = starts(H, TILE_MAX_SIDE, TILE_OVERLAP)
    tiles_x = starts(W, TILE_MAX_SIDE, TILE_OVERLAP)

    all_masks: List[Dict[str, Any]] = []
    for y0 in tiles_y:
        for x0 in tiles_x:
            y1 = min(H, y0 + TILE_MAX_SIDE)
            x1 = min(W, x0 + TILE_MAX_SIDE)
            tile = image[y0:y1, x0:x1]
            tmasks = mask_generator.generate(tile)
            for m in tmasks:
                # shift bbox to global coordinates
                bx, by, bw, bh = m.get("bbox", [0, 0, 0, 0])
                m["bbox"] = [int(bx + x0), int(by + y0), int(bw), int(bh)]
                all_masks.append(m)
    # Deduplicate overlapping masks across tiles
    return dedup_by_iou(all_masks, thr=0.7)
