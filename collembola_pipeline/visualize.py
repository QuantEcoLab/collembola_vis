from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .config import OVERLAY_MIN_LABEL_CONF


def draw_overlay(
    image: np.ndarray, masks: List[Dict[str, Any]], rows: List[dict], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image)
    # Accepted organism rows include bbox and p_collembola
    for r in rows:
        x, y, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
        rect = Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        p = float(r.get("p_collembola", 0.0))
        if p >= OVERLAY_MIN_LABEL_CONF:
            ax.text(
                x,
                max(0, y - 3),
                f"{r.get('organism_id', '')}:{p:.2f}",
                color="lime",
                fontsize=7,
                weight="bold",
            )
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
