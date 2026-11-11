# Experimental exp03_no_os application integration

Branch: **`feature/integrate-exp03-no-os`**. No commit/push or production model
replacement is part of this integration.

## Preservation and checkpoint

The original branch was `feature/ruler-contour-correction`, HEAD
`da4be9931ae5ca5d44bbf31893af233613544467`. Before switching, all 50 dirty/untracked
user files were copied with binary diffs, a tar archive and hashes to:

`outputs/integrate_exp03_worktree_backup_20261104T162248086371Z/`.

The only segmentation checkpoint used by the new contour workflow is:

`outputs/ruler_v7_reviewed367_ft_exp03_no_os/evaluation/fixed_candidate/best.pt`

SHA-256:
`3fe97dd43f387815f8ee947eb5d02a6b0a7afadc1be59942a6b4a1f49f9aa0da`.

It is checked before the local launcher imports the app, again during FastAPI
startup, on submission, and before/after neural processing. A mismatch stops the
feature; there is no fallback model.

## Workflow and unchanged inference

The final control in `WorkspaceSidebar` is **Napravi konture**, below the existing
Reset control and every existing result/control. Its model label explicitly says
experimental `exp03_no_os`. Progress/errors appear above the button. Both immediate
submission and pending/running states disable it; duplicate API submissions for
the same image reuse an active job.

`POST /api/contours/run` accepts only `image_id`, optional `detection_job_id`,
optional `source_contour_job_id` (measurement-only), and nullable `um_per_pixel`.
Inference overrides such as confidence, model path or padding are rejected.

- Existing `backend.services.detection.run_detection` / `scripts.infer_tiled`
  supplies tiled detection, or a completed, same-image standard detector job is reused.
- Detector: existing `models/yolo11n_tiled_best.pt`; tile 1280, overlap 256,
  confidence 0.6, global bbox NMS IoU 0.5. CPU execution follows the prior fixed
  qualitative pipeline and uses four torch threads in the local launcher.
- Crops: original floor/ceil, photo-edge clipping, **zero extra padding**.
- Segmentation: same legacy `result_mask` with imgsz320, confidence0.15 and CPU.
- Dedup: unchanged `duplicate_pair_classifier_v7_fragment_aware.joblib` and
  `apply_classifier_dedup`, threshold0.50. No new morphology, area cleanup,
  envelope or mask-IoU dedup is introduced.
- Only the segmentation checkpoint is replaced. Output files live in the configured
  web job directory. The workflow does not call fine-tuning or annotation-save code.

Results include the final overlay, per-instance masks/contours, detector/dedup
hashes and parameters, dedup audit, structured measurements, CSV and Excel.

## Existing measurement definitions and units

`scripts.measure_organisms.measure_organism_from_mask` is reused directly on each
nonempty **dedup-kept** exp03 mask. The method's axes, area, perimeter and cylinder
formula are unchanged. The SAM imports were made lazy so measuring an existing mask
does not load SAM or require its checkpoint. No threshold-based resegmentation or
empty-mask bbox fallback is used for experimental measurements.

- Length = existing regionprops major axis of the equivalent ellipse.
- Maximum transverse width **of that equivalent ellipse** = existing minor axis.
  This is clearly labelled in the table; it is not a new Feret/cross-section width.
- Area and perimeter = the existing binary-mask region properties.
- Estimated volume = existing `V = π × (width / 2)² × length`.
- Existing µm values are converted to mm / mm² / mm³ by factors 1e3 / 1e6 / 1e9.
- Volume is an estimate, **not a directly measured 3D volume**.

All retained selected-mask pixels are measured, including disconnected components
when present; multi-component and crop-edge warnings make that limitation visible.
Discarded duplicates and missing/empty masks never enter measurements or volume
statistics. Degenerate axes have no physical volume value.

Calibration must be finite, positive and confirmed for the current image. The
historical global 8.57 preset is not automatically accepted for a different image.
Users can explicitly type/select a known µm/px value or use the existing two-point
0–10 mm ruler calibration. Image-specific values persist by image ID.

Without calibration, measurements remain in pixels; physical columns and volume
are null, and the UI states **nije dostupan bez kalibracije**. CSV/Excel include
the `volume_mm3` and `volume_unit` fields even when unavailable. The volume summary
contains count, mean, median, minimum, maximum and total; it does not invent a
physical zero when calibration is absent.

Changing calibration submits a measurement-only job using saved masks, without
rerunning detector, segmentation or classifier. Old physical results and export
are held back until the new calibration's result is ready. Clicking a table row
maps its detection ID (not row position) to the corresponding contour, highlights
it and focuses the image; the results dialog also shows the selected contour.

## Local preview, isolated from original app/experiment data

Local URL: **http://127.0.0.1:8774/workspace**.

The launcher sets all writable app directories to
`outputs/integration_exp03_local/app_data/`, including its own community/project
database, uploads, annotations and calibration directory. Existing experiments,
datasets, review databases, original annotations, source photos and model weights
are read-only inputs. Existing app authentication is used (e.g. the existing
`user1` account); no auth bypass or new default account was introduced.

The real browser smoke test created an isolated **Exp03 integration preview**
project, with the already qualitative-reviewed source photograph and a complete
pixel-only exp03 measurement result. Its IDs/URL are in
`outputs/integration_exp03_local/browser_smoke.json`. No calibration or GT was
invented for that photograph.

```bash
# Dependencies are isolated under the new local preview directory.
.venv-ruler-ft/bin/python -m pip install \
  --target outputs/integration_exp03_local/runtime_dependencies \
  -r backend/requirements-exp03.txt

# From repository root; uses Node 22 and creates a uniquely versioned build:
python3 -B scripts/build_local_exp03_frontend.py

# From repository root; validates branch and checkpoint before app startup:
.venv-ruler-ft/bin/python -u -B scripts/run_local_exp03.py --port 8774
```

The normal Vite default remains `/collembola/`; only the local preview build sets
`VITE_BASE_PATH=/`. Backend production dist defaults remain unchanged. The local
launcher selects its isolated dist through `COLLEMBOLA_FRONTEND_DIST`.

The build helper atomically selects `outputs/integration_exp03_local/active_frontend.json`.
The launcher verifies every generated file hash and refuses to fall back to the old
unversioned dist. A single Uvicorn process serves both API and frontend on port 8774.
The visible bottom-right local-preview badge reports build ID, UTC timestamp,
HTML/JavaScript agreement, API build agreement and server PID. `/api/preview-build`
provides the same identity independently. Asset URLs include the unique build ID.

Local preview HTML/API/assets use no-store headers. On startup the browser removes
same-origin service-worker registrations and Cache Storage entries, with a bounded
two-second cleanup deadline. HTML responses clear HTTP cache only. Auth, workspace
and calibration localStorage are preserved. A local-only auth signing key persists
between launcher restarts; no database or project reset is involved.

## Tests and verification

```bash
PYTHONPATH=outputs/integration_exp03_local/runtime_dependencies:. \
  .venv-ruler-ft/bin/python -B -m pytest backend/tests -q

# From frontend/ with Node 22:
npm test
npm run build
```

Backend tests cover request/error/hash checks, standard detector reuse, duplicate
submission, exact mask axes and volume/unit conversion, exclusion of duplicates
and missing masks, pixel-only behavior, degenerate masks, CSV/Excel output,
measurement-only reuse, fixed inference kwargs/crops and startup failure before
database initialization. Frontend tests cover last-button position, busy/error
states, image-required behavior, real API payload shape, per-image calibration,
automatic results, stable detection-ID selection, raw/contour toggling and summary
statistics.

The real headless Chrome smoke test exercises normal login, server-path image
loading, missing-image feedback, one job on double-click, real exp03 inference,
automatic measurement of 1036 final masks, no-calibration volume handling,
CSV/Excel downloads and correct contour selection. It reported zero JavaScript
errors. Synthetic calibration tests verify physical volumes without assigning an
unproven calibration to the real photograph.
