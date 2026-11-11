# Curated detector corrections — local feature preview

Branch: `feature/curated-detector-dataset`. No training, model replacement, commit
or push is performed by collection or export.

## User workflow

1. Load an image in Workspace and optionally run the existing detector.
2. Open **Edit Detections**. Use **Invalid** to reject false positives, **Draw** to
   add missed organisms. Select an existing box and edit x1/y1/x2/y2 in pixels,
   then **Primijeni koordinate**. The same detection ID is retained.
3. In the left sidebar, enter the source/experiment note and click
   **Spremi za budući trening**. This saves the current in-memory edits directly;
   a separate ordinary annotation Save is not required.
4. Open **Dataset za trening** to inspect versions, originals, overlays, per-box
   added/corrected/confirmed status, rejected detections and provenance, and
   download **Izvezi YOLO ZIP**. Empty final sets are reviewed negative images.

## Storage and immutability

Default storage is `data/curated_detector`; the local preview explicitly uses
`outputs/integration_exp03_local/app_data/curated_detector`.

### Legacy detection-job identity compatibility

The pre-existing single-image detection route and batch sub-jobs persisted
`image_path`, but not `image_id`. Collection must not mistake an absent field for
a contradictory photo ID. Legacy jobs are accepted only after their registered
upload path (including image-ID directory and filename), result path, detection
metadata image path and native dimensions agree with the current registered
photograph. Any explicitly contradictory ID/path/dimensions still blocks saving;
no alternative job is guessed and no old job/annotation file is rewritten.

Collection records the verified current byte SHA256 and whether a historical
hash was available. Legacy jobs have no historical image hash; it is not
fabricated retroactively. New single-image and batch jobs capture image ID,
filename, dimensions and SHA256 before inference, check it after inference and
persist the identity. Collection compares that captured identity when present.

The AS_MOS_Ctrl_4_004 recovery preserved the saved annotation JSON and original
photo before changing code. Its 661 accepted + 33 rejected + 1 added boxes were
saved without changing any submitted IDs, coordinates, confidences or statuses.
The frontend build was left in place so the open Workspace required no reload.
Audit evidence: `outputs/integration_exp03_local/as_mos_save_verification.json`.

- `objects/<SHA256>/original.<ext>` contains a byte-for-byte copy, not a symlink.
- `versions/vNNNNNN-<id>/manifest.json` is a complete immutable dataset snapshot.
- `seal.json` verifies the snapshot's SHA256 before browsing or exporting.
- Corrections to the same file/pixel identity create a new revision/version while
  retaining earlier versions. A version contains one entry per source image.
- Identical resubmission is idempotent. A process-safe file lock serializes writers;
  staging plus atomic rename prevents partially visible versions.
- Manifests retain source/registered and resolved paths, image dimensions, EXIF
  capture date when present, save timestamp/user, original detector job and CSV
  hash, original boxes, submitted/rejected boxes, final boxes, removed detections,
  derived edit classifications and split-registry evidence.

Existing photographs, annotations, datasets and models are read-only inputs to
this feature. The original photo hash and modification timestamp are checked
again before publishing a version. Authentication remains required for browsing,
images, saves and export.

## Split leakage checks and scope

Prepare the read-only registry before saving:

```bash
PYTHONPATH=outputs/integration_exp03_local/runtime_dependencies:. \
  .venv-ruler-ft/bin/python -B scripts/build_detector_split_registry.py \
  --root /absolute/path/to/repository \
  --root /absolute/path/to/other/historical/datasets \
  --output outputs/integration_exp03_local/detector_split_registry.json
```

For a normal backend set `COLLEMBOLA_DETECTOR_SPLIT_REGISTRY` and optionally
`COLLEMBOLA_DETECTOR_DATASET_DIR`. The local launcher sets these automatically.
The registry inventories configured roots, split image directories, source/split
metadata, train/val CSVs, split image lists and detector plate declarations.
Original sources are grouped with their crops/tiles. Matching uses normalized
source identity, SHA256 of bytes and native/EXIF-oriented decoded RGB pixels,
without resizing. Thus a rename or lossless re-encoding cannot bypass membership
for a registered source photograph. Arbitrary edited/lossy image derivatives are
not claimed to be perceptual matches.

The dataset page shows the registry roots/date/counts. This is a check against
those configured existing datasets, not undiscovered remote or deleted datasets.
Missing registry, missing root, or changed membership-evidence inventory blocks
saving until the registry is rebuilt. App uploads, this new collection, frontend
builds, dependencies and worktree backups are excluded from split evidence.

Final boxes must have unique IDs, finite coordinates, positive area and lie
strictly within image bounds (right/bottom may equal width/height). Exact and
near-identical duplicates with IoU >= 0.95 are rejected rather than silently
deleted. Rejected detections stay in the audit, not in final labels. Original
accepted/corrected boxes are checked against the same-image completed detector
CSV; unknown boxes must be explicitly added.

## YOLO export

Export is an explicit authenticated download; it does not submit any job.

- `images/train/*.jpg`, `labels/train/*.txt`: class 0, normalized YOLO detection
  `center_x center_y width height`.
- Existing edge-aligned 1280 tiles, overlap 256 / stride 1024, with the existing
  `map_roi_to_tile` 30% box-retention rule. Small photos stay native-sized and
  labels are normalized to actual tile dimensions. Reviewed background tiles
  are included, unlike the old positive-only dataset preparation default.
- `originals/` and `original_labels/` retain full-photo bytes and normalized labels.
- `manifest.json`, `tile_metadata.json`, `README.txt` retain source grouping/audit.
- `data.yaml` declares one `collembola` class and train image folder, with `val:null`.
  This is a future-training pool, not an automatically split ready-to-train run.
  An independent validation set must be deliberately configured before a future
  training command. Never split tiles from one photograph across partitions.

## Development

The existing immutable preview build helper and launcher accept this branch:

```bash
python3 -B scripts/build_local_exp03_frontend.py
.venv-ruler-ft/bin/python -u -B scripts/run_local_exp03.py --port 8774
```

Preview: `/workspace` and `/detector-dataset` on the same port 8774. Build identity
and frontend/API agreement remain visible in the bottom-right preview badge.
