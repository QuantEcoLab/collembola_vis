# Ruler / v7 Workflow and Provenance

## Scope and external artifacts

This branch preserves the ruler/v7 tools and small provenance reports recovered
from the verified `ruler-v7-20261027T162453Z` snapshot, plus the local PNG-only
crop selector. The snapshot itself, source images, LabelMe polygon files, YOLO
datasets, masks, generated comparisons and model weights are external artifacts.
No training or dataset generation was run while preparing this branch.

Scripts are under `scripts/` and `scripts/ruler_v7/`; versioned manifests and
review decisions are under `reports/ruler_v7/`. Historical paths in those reports
still refer to external artifacts under `data/` and `runs/`. Only personal
absolute Windows checkout prefixes were removed from report copies. Relative
path suffixes, separators, source/detection IDs, timestamps, decisions and metrics
were preserved. Readers normalize Windows separators when opening paths on Linux.

## Dataset and leakage limitations

**The six manually annotated ruler scenes are NOT an independent held-out set.**
They come from three source photographs, all of which contributed v7 training
crops:

- `C_2_1_Collembola007`
- `C_4_3_PS_500034`
- `K_3_Collembola003`

The six reviewed selections contain 129 polygons. The instance-crop generator
accepted 83 candidates (80 unchanged, 3 adjusted) and rejected 46. It targets
65-85% body-bbox occupancy (75% nominal) and rejects cuts through annotated
neighbours. It cannot establish the absence of unannotated organisms: visual
edge inspection remains necessary. The historical v7 dataset has 1,914 train
image/label pairs and 463 validation pairs; the large ruler scenes were not added
directly as train images. See `reports/ruler_v7/v7/` and
`reports/ruler_v7/ruler/export_summary.csv`.

The final clean-evaluation subset retained 259 of the 463 validation images and
excluded 204. Of 57 possible natural-edge-contact cases, manual review retained
50 complete organisms and excluded 7 truncated ones. This is a cleaned validation
subset, not a new independent ruler test set. See `reports/ruler_v7/clean_eval/`.

## Models and recorded evaluation

- Detector: `models/yolo11n_tiled_best.pt`.
- v5 segmentation reference: `runs/segment/runs/segment/trunk_seg_v5_test20/weights/last.pt`.
- v7 ruler segmentation: `runs/segment/runs/segment/trunk_seg_v7_ruler_finetune/weights/best.pt`.
- v7 fragment-aware duplicate classifier (a different model from the v7
  segmentation model): `models/duplicate_pair_classifier_v7_fragment_aware.joblib`.

All weights are external; no checkpoints are added by this branch. Recorded
checkpoint hashes and sizes are in
`reports/ruler_v7/v7/ruler_heldout_provenance_summary.json`.
`reports/ruler_v7/models/v7_training_args.yaml` is a historical configuration
record, including its resume setting; it is not a fresh-training recipe.

`reports/ruler_v7/evaluation/clean_eval_v7_final_metrics_clean.csv` preserves the
recorded aggregate evaluation: mask mAP50-95 is 0.7798883743699675 for v5_last and
0.7732895776654839 for v7_best. These are not ruler-only held-out metrics.
The visual review is also not an unbiased overall accuracy estimate: it
prioritizes missing masks and large disagreements. See the manual-review document
for its scope and counts; do not infer a production-model winner from it alone.

## Linux environment and crop selection

Run commands from the repository root in the project Python environment. The
PNG-only selector uses Pillow and Tkinter, with no OpenCV dependency. On
Debian/Ubuntu, system Python's Tk support is provided by `python3-tk` and
`python3-pil.imagetk`. The recovered LabelMe/comparison tools additionally use
OpenCV, NumPy, pandas, joblib, PyTorch and Ultralytics. Existing project
requirements cover most of these; use an environment with the appropriate GUI
packages installed. GUI tools require a desktop display.

For a new, unannotated PNG crop (PNG and metadata only, no LabelMe JSON):

```bash
python3 scripts/select_ruler_crop_review.py --image "path/to/source.png"
```

Drag a rectangle; Enter saves; C clears; Q/Esc exits. Open the saved PNG
separately in LabelMe and annotate every complete visible organism. Existing
source images must be restored separately; these commands do not recover images.

For a crop that already has LabelMe polygons:

```bash
python3 scripts/ruler_v7/select_ruler_crop_review.py --labelme-json "path/to/annotation.json"
python3 scripts/ruler_v7/select_all_ruler_crops_review.py
```

The recovered LabelMe selector uses Enter to preview, S to save, R to repeat,
and left-click markers for unannotated cut organisms (right-click undoes a
marker). These markers are manual, not automatic detections. The batch launcher
uses `reports/ruler_v7/ruler/crop_metadata.csv`, containing six active crops. The
dense `C5_2_Fe2O3003_ruler_dense` crop remains separately listed as excluded.
Both selectors write new selections under the ignored `data/` tree by default.

## Reproduction boundaries

- `scripts/generate_trunk_dataset_v7_instance_crops.py` reconstructs the recorded
  instance-crop addition when the external v5 and ruler-export datasets exist.
  Running it writes a new dataset and refuses an existing v7 directory; it is
  not a read-only review command.
- `scripts/review_possible_natural_edge_contact.py` uses the recorded clean-eval
  manifest and external validation images/labels. Interactive decisions update
  review CSVs only when the reviewer runs it.
- The comparison and raw-mask review commands are documented in
  `V7_RULER_FINAL_PIPELINE.md` and `V7_RULER_MANUAL_REVIEW.md`.
- The snapshot does not provide standalone scripts for every earlier
  LabelMe-to-YOLO export or final clean-eval construction step. Their manifests
  document the result but do not, by themselves, reproduce the missing steps.
- Python syntax validation does not establish numerical/model reproducibility.
  Record runtime package versions and restore the matching external artifacts
  before reproducing any inference or evaluation.

## Historical initial comparison status

The following note records an earlier blocked phase. Later final-pipeline and raw
review artifacts were present in the snapshot, as documented in the companion
documents; the blocked status below is not the current status of all v7 work.

Status: blocked before inference.

Leakage-free full-resolution candidates with bright ruler/scale were found and selected from new source IDs. None of the selected source IDs appear in `data/trunk_dataset_combined_v7/yolo_seg`, and none are one of `C_2_1_Collembola007`, `C_4_3_PS_500034`, or `K_3_Collembola003`.

Inference was not run because the existing production tiled-inference and dedup implementation with exact tile dimensions, overlap, confidence/IoU thresholds, preprocessing, deduplication and postprocessing parameters was not found in the workspace. Running a recreated pipeline would not satisfy the requirement to use the exact same existing production procedure.

Files:
- `candidate_review/leakage_free_fullres_candidates.jpg`: visual review sheet for leakage-free full-resolution candidates.
- `candidate_selection_leakage_audit.csv`: selected candidate images and source-level leakage check.
- `blocked_summary.json`: machine-readable blocked status.
