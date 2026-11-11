"""Which boxes a photo uses, and merging a newer detection job into reviewed annotations.

An annotation file is *human-saved* when it has ``saved_at``: the save endpoint has always set it.
Files without it were written from raw detections by older batch code (e.g. at project import) and
hold no human decision; their boxes are byte copies of the detection job named in ``source_job_id``.

Rules:
* a human-saved file always wins over detection jobs (it holds the reviewer's work);
* a file without ``saved_at`` only wins while no other detection job is current for the photo: a
  newer job (e.g. a re-run with the current detector) replaces it;
* ``merge_detections`` replaces the model boxes of a reviewed file with the boxes of a newer job
  and keeps the reviewer's decisions: every added and rejected box stays, a new box on an object
  the reviewer rejected or drew is dropped, and a previously accepted box that no new box covers is
  kept (a reviewer looked at it).
"""

import json
from pathlib import Path

import pandas as pd

from backend.config import settings

MATCH_IOU = 0.3
CONTAIN = 0.8


def annotation_path(image_id: str) -> Path:
    return settings.annotations_dir / f"{image_id}.json"


def read_annotation(image_id: str) -> dict | None:
    """The saved annotation file of a photo, or None if missing, unreadable or for another photo."""
    path = annotation_path(image_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("image_id") != image_id:
        return None
    return data


def is_human_saved(data: dict | None) -> bool:
    return bool(data and data.get("saved_at"))


def annotations_take_precedence(data: dict | None, detection_job_id: str | None) -> bool:
    """Whether a photo's boxes come from its annotation file rather than ``detection_job_id``."""
    if not data or not data.get("boxes"):
        return False
    if is_human_saved(data) or not detection_job_id:
        return True
    return data.get("source_job_id") == detection_job_id


def detection_job_image_id(job) -> str | None:
    """Photo a detection job belongs to (older jobs: the upload folder of their image path)."""
    result = job.result or {}
    image_id = result.get("image_id") or (result.get("image_identity") or {}).get("image_id") or (job.params or {}).get("image_id")
    if not image_id and result.get("image_path"):
        image_id = Path(result["image_path"]).parent.name  # data/uploads/<image_id>/<file>
    return image_id


def detection_job_boxes(job) -> list[dict]:
    """Boxes of a completed detection job in the annotation-file format (status accepted)."""
    csv_path = (job.result or {}).get("csv_path")
    if not csv_path:
        raise FileNotFoundError("No CSV result for this job")
    df = pd.read_csv(csv_path)
    return [{"id": f"box-{i}", "x1": float(row["x1"]), "y1": float(row["y1"]), "x2": float(row["x2"]),
             "y2": float(row["y2"]), "conf": float(row.get("confidence", row.get("conf", 1.0))), "status": "accepted"}
            for i, row in df.iterrows()]


def _iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def _inside(a: dict, b: dict) -> float:
    """Fraction of box a lying inside box b."""
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    area = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    return ix * iy / area if area > 0 else 0.0


def _covers(new: dict, old: dict) -> bool:
    return _iou(new, old) >= MATCH_IOU or _inside(old, new) >= CONTAIN or _inside(new, old) >= CONTAIN


def _replaces(new: dict, old: dict) -> bool:
    """A new box takes the place of a previously accepted box when it matches it or contains it.
    A smaller new box lying inside it (a partial box) does not displace the reviewed box."""
    return _iou(new, old) >= MATCH_IOU or _inside(old, new) >= CONTAIN


def _partial_of(new: dict, old: dict) -> bool:
    """A new box that is only part of a previously accepted (reviewed) box."""
    return _inside(new, old) >= CONTAIN and not _replaces(new, old)


def merge_detections(annotation: dict | None, detections: list[dict]) -> tuple[list[dict], dict]:
    """Replace the model boxes of ``annotation`` with ``detections``, keeping the reviewer's decisions.

    Returns the merged boxes and a summary with counts. Nothing is written.
    """
    old = list((annotation or {}).get("boxes") or [])
    human = is_human_saved(annotation)
    added = [b for b in old if b.get("status") == "added"]
    rejected = [b for b in old if b.get("status") == "rejected"]
    accepted = [b for b in old if b.get("status") == "accepted"] if human else []

    new, dropped_rejected, dropped_added, dropped_partial = [], 0, 0, 0
    for box in detections:
        if any(_iou(box, r) >= MATCH_IOU for r in rejected):
            dropped_rejected += 1
        elif any(_covers(box, a) for a in added):
            dropped_added += 1
        elif any(_partial_of(box, a) for a in accepted):
            dropped_partial += 1  # the reviewed box of that animal stays
        else:
            new.append({**box, "status": "accepted"})

    def carried(box):
        box = dict(box)
        if not str(box.get("id", "")).startswith(("added-", "prev-")):
            box["id"] = f"prev-{box.get('id', '')}"
        return box

    kept_accepted = [carried(b) for b in accepted
                     if not any(_replaces(n, b) for n in new) and not any(_covers(a, b) for a in added)]
    boxes = new + [carried(b) for b in rejected] + added + kept_accepted
    summary = {
        "human_saved": human,
        "new_detections": len(detections),
        "new_kept": len(new),
        "new_dropped_on_rejected": dropped_rejected,
        "new_dropped_on_added": dropped_added,
        "new_dropped_inside_previous": dropped_partial,
        "previous_accepted": len([b for b in old if b.get("status") == "accepted"]),
        "previous_accepted_kept": len(kept_accepted),
        "added_kept": len(added),
        "rejected_kept": len(rejected),
    }
    return boxes, summary
