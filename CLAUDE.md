# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO-based detection and measurement pipeline for collembola organisms in ultra-high-resolution microscope images (~10K×10K pixels). Uses tiled inference instead of downscaling to preserve detail, achieving 99.2% mAP@0.5. Licensed AGPL-3.0 (required by Ultralytics dependency).

A full web UI wraps the pipeline: users upload images, run detection, edit annotations, measure organisms, and share results via a collaborative database.

## Environment Setup

```bash
conda create -n collembola python=3.11
conda activate collembola
pip install -r requirements.txt
```

No formal test suite exists. Validation is done manually by running scripts on sample data and inspecting overlay visualizations.

## Key Commands

### Tiled Inference (primary use case)
```bash
python scripts/infer_tiled.py \
    --image "data/slike/IMAGE.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --conf 0.6 --device 0
```

### Measurements
```bash
# Fast ellipse method (178 org/sec, recommended)
python scripts/measure_organisms_fast.py \
    --image "data/slike/IMAGE.jpg" \
    --detections infer_tiled_output/IMAGE_detections.csv \
    --um-per-pixel 8.57

# SAM-based method (1 org/sec, more accurate contours)
python scripts/measure_organisms.py \
    --image "data/slike/IMAGE.jpg" \
    --detections infer_tiled_output/IMAGE_detections.csv \
    --um-per-pixel 8.57 --device cuda
```

### Training
```bash
python scripts/convert_imagej_rois.py          # Step 1: ROIs → CSV
python scripts/create_tiled_dataset.py          # Step 2: CSV → tiled YOLO dataset
python scripts/train_yolo_tiled.py \            # Step 3: Train
    --device 0,1,2,3 --epochs 100 --batch 32 --patience 30
```

### Batch Processing
```bash
python scripts/process_plate_batch.py \
    --images "data/slike/*.jpg" \
    --model models/yolo11n_tiled_best.pt \
    --um-per-pixel 8.57 --output-dir outputs/batch
```

## Architecture

### Data Flow
```
ImageJ ROIs → CSV → Tiled Dataset → YOLO Training → Model (.pt)

Large Image → Tiled Inference → Detections CSV → Measurements → Final CSV
                                      ↓
                              Annotation Editing → Community DB
                                      ↓
                                Overlay Visualization
```

### Tiled Processing Pattern (core abstraction)
Images are split into 1280×1280 tiles with 256px overlap (stride=1024). Each tile is processed independently by YOLO, then results are merged using global NMS (IoU=0.5) to deduplicate detections at tile boundaries. This pattern appears in both inference (`scripts/infer_tiled.py`) and training data preparation (`scripts/create_tiled_dataset.py`).

### Directory Layout
- **`scripts/`** — Standalone CLI tools (argparse-based). Each script is a complete entry point.
- **`collembola_pipeline/`** — Reusable Python package. `config.py` is the central configuration with all thresholds, paths, and SAM parameters.
- **`models/`** — Pre-trained YOLO model (`yolo11n_tiled_best.pt`, 5.4 MB).
- **`data/`** — Images, annotations, datasets (gitignored).
- **`data/annotations/`** — Per-image annotation JSON files (`{image_id}.json`), saved from the web UI.
- **`data/community.db`** — SQLite database for collaborative submissions.
- **`archive_unused/`** — Previous approaches (classical CV + SAM proposals) kept for reference.

### Two Detection Approaches
1. **YOLO tiled inference** (production) — `scripts/infer_tiled.py` handles the full tile→detect→NMS pipeline.
2. **Classical CV + SAM** (archived in `collembola_pipeline/`) — Region proposal → CNN classification → segmentation refinement. The `proposal_*.py`, `classify*.py`, and `segment*.py` modules implement this older pipeline.

### Measurement Methods
Both methods take YOLO detection boxes and produce per-organism measurements (length, width, area, volume). Volume uses a cylinder model: `V = π × r² × h`. The fast method fits ellipses via eigenvalue decomposition on binary masks; the SAM method generates precise contours.

When `use_annotations=True` is passed to the measurement endpoint, the backend reads the saved annotation JSON for that image, filters out rejected boxes, and generates a temporary detections CSV. This allows measurements to reflect manual edits.

### Web UI
- **`backend/`** — FastAPI app. Run with `uvicorn backend.main:app --reload`.
- **`frontend/`** — React + Vite + TypeScript + Tailwind CSS. Run with `cd frontend && npm run dev`.
- Deployed with `root_path="/collembola"` behind a reverse proxy.

### Web UI Pages
- **`/`** — Workspace: upload image → run detection → edit annotations → measure → export.
- **`/collaborate`** — Collaborate: browse community submissions, load boxes from another user's analysis, submit your own results.

### Web UI Development
```bash
# Install dependencies (one-time)
make install

# Run both servers (backend :8000, frontend :5173)
make dev

# Or run individually
make backend
make frontend

# Production build (served from backend's /files/dist)
cd frontend && npm run build
```

The frontend proxies `/api`, `/ws`, and `/files` to the backend via Vite config.

### Auth & Roles
- JWT-based auth (`backend/auth.py`). Roles: `user`, `admin`.
- Default accounts: `user1/user12345` (user), `admin/admin12345` (admin).
- Fine-tuning panel is admin-only. All other features available to all authenticated users.
- Login redirects to `/collaborate`.

### Annotation Editing (Edit Detections mode)
Boxes have three statuses:
- **`accepted`** — original YOLO detection, kept (green overlay)
- **`rejected`** — marked invalid by user, excluded from measurements but preserved for fine-tuning (red overlay, cross-out)
- **`added`** — manually drawn by user (blue dashed overlay)

Boxes with IoU ≥ 0.5 against another non-rejected box are highlighted orange as suspected duplicates.

Annotations are saved to `data/annotations/{image_id}.json` and auto-restored when the same detection job is loaded.

### Collaborate / Community DB
- `backend/db.py` — SQLite with `community_detections` table.
- `backend/routers/community.py` — REST endpoints: submit, list, stats, get by id, export (JSON/CSV).
- Submissions are **upserted** keyed on `(username, image_id)` — re-submitting the same image updates the existing record rather than creating a duplicate.
- `frontend/src/store/communityLoadStore.ts` — in-memory store for boxes pending load into workspace.

### Frontend State Persistence
- `frontend/src/store/workspaceStore.ts` — persists `image`, `detectionJobId`, `measureJobId` to localStorage so workspace survives page refresh.
- `frontend/src/store/calibrationStore.ts` — persists `umPerPixel` across sessions.
- `frontend/src/store/authStore.ts` — persists JWT token and role.

## Conventions

- Scripts follow `verb_noun.py` naming (e.g., `infer_tiled.py`, `measure_organisms_fast.py`)
- All scripts use argparse with `if __name__ == '__main__': main()` pattern
- Output files: `{image_stem}_{suffix}.{ext}` with JSON metadata alongside
- Calibration (`--um-per-pixel`) is critical — all physical measurements depend on it
- GPU is auto-detected; all scripts fall back to CPU
- PEP 8 style; docstrings on main functions
- Frontend: Zustand for global state, TanStack Table for measurement display, React Router v6
