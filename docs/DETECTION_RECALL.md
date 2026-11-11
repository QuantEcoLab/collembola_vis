# Detection recall on Pendifin plates (detector v1, November 2026)

## Problem

On the Pendifin Čačinci (CAC) and Erdut (ERD) scans the production detector
(`models/yolo11n_tiled_best.pt`, conf 0.6, 1280 tiles) misses many animals. The
substrate is mottled grey and many animals look slightly hazy, so the model
scores them at 0.2-0.6. Lowering the threshold alone does not help: the extra boxes
are mostly substrate texture, grid paper lines, the dish rim, fibres and fragments
of animals that already have a box.

Three things were missing:

* small juveniles. At native scale the model scores them low; on 640 px tiles,
  which the model sees upscaled 2x, it finds them with high confidence;
* hazy medium and large adults on CAC plates (C1, C3, Ctrl) that the model never
  saw in training;
* fragment duplicates. A small box inside the box of the same animal costs
  precision, and NMS keeps it because the IoU is small.

## Standard detection profile (backend `settings`)

| setting | value |
|---|---|
| `default_model` | `models/yolo11n_tiled_pendifin_v1.pt` (see below; not in git) |
| `default_conf` | 0.4 on 1280 tiles, overlap 256 |
| `default_iou` | 0.5 global NMS |
| `default_contain` | 0.8: drop a box that lies >= 80% inside a larger kept box (fragment duplicates) |
| `default_extra_tile_sizes` / `default_extra_conf` | `[]` / `[]`: no extra pass by default |

`scripts/infer_tiled.py` takes `--contain 0.8` and optional extra passes, for example
`--extra-tile-sizes 640 --extra-conf 0.8`, which run 640 px tiles seen 2x upscaled.
Without these flags it behaves as before.

Why there is no 640 pass by default: with the base model it was the cheapest
recall lever, because it finds small juveniles. With the fine-tuned model it added
1 of 101 dev animals. On replicate-3 images, more than half of the boxes it added
were dish-rim speckle, fibres or fragments. It also costs about 4x the tiles.

Each detection job now records the resolved settings as `result.detector_profile`.
The exp03 contour router accepts detections that match either the current standard
profile or the legacy one (old model, conf 0.6, 1280 tiles, no fragment
suppression), so older jobs stay usable. A job recorded before profiles existed
counts as legacy. The exp03 contour job's own detector run still uses the legacy
detector it was validated with (`settings.legacy_model`).

The workspace's standard "Run/Re-run Detection" sends no thresholds. The advanced
dialog sets the confidence. Fragment suppression stays on.

## Deployment

Copy the weights to `models/yolo11n_tiled_pendifin_v1.pt` on the server before
restarting the service. SHA-256:
`ac3d0b995a57be3109b31e5958e042c6b368c75b2ab6d45407ce021568b5b649` (5,624,211 bytes;
the v1 build is at `adeb:/home/adeb/agent-work/developer-recall/models/`). Keep
`models/yolo11n_tiled_best.pt`, which is the legacy detector. Speed is unchanged:
it is the same network size on the same tiles.

## How the training labels were built

Only TRAIN-split images were used: replicates 1-3 of both projects, minus
`CAC_PDF_Ctrl_2_002` (the user-flagged evaluation image). Holdout replicates 4-5
were never read for labels.

1. Candidates. The production model was run at conf 0.05 on 1280 tiles and on
   640 tiles, then merged with NMS 0.5. Candidates matching an annotation box
   (IoU >= 0.3) were removed, and so were candidates >= 80% inside a larger box.
   The remaining extras with conf >= 0.25 on the 17 CAC train images were reviewed.
2. Bright context means the median grey of a 3x box window is > 110: grid paper,
   ruler bars and the dish rim. A seeded check showed 70/70 of these were not
   animals. They were labelled negative automatically.
3. Every remaining extra (1,842) was judged visually on CLAHE chip sheets at
   about 1.3x. Each was labelled animal, uncertain, or not (texture, fibre, rim,
   fragment). Results: 782 animals, 256 uncertain. They are in
   `scripts/recall_ft/verified_extras_v1.json`. The share of animals per conf band
   (fold A): 0.25-0.35: 16%, 0.35-0.45: 23%, 0.45-0.6: 42%, 0.6-0.8: 71%, >= 0.8: 84%.
   No single threshold would have given clean labels.
4. The labels are annotation boxes (accepted + added) plus the verified animals.
   Fragments (>= 80% inside a larger box) are dropped. Uncertain objects are
   inpainted. ERD images use the human review as is.
5. Tiles are 1280 with stride 1024, plus 30% 2x-upscaled 640 crops and 35% of the
   empty tiles as hard negatives. The original training set (`data/yolo_tiled`)
   is mixed in so other plates keep working
   (`scripts/recall_ft/build_dataset.py`).
6. The model was fine-tuned from `yolo11n_tiled_best.pt` for 25 epochs at imgsz
   1280, batch 4, SGD lr0 0.003, mosaic, and flips in both directions
   (`scripts/recall_ft/train_ft.py`).

The thresholds were tuned with a model trained only on replicates 1-2. It was scored
on replicate-3 images: 20 dev tiles of 1024 px on CAC that I labelled exhaustively
(101 animals), plus the human-reviewed ERD replicate-3 images.

Results with the model trained on replicates 1-2, scored on replicate 3 (conf 0.4,
fragment suppression on, no extra pass). The production settings are shown for
comparison:

| | my CAC dev tiles (101 animals) | ERD rep-3 human review (943 GT) |
|---|---|---|
| production (conf 0.6) | R 0.752, P 0.905 | R 0.898, P 1.000* |
| fine-tuned, conf 0.4 | R 0.970, P 0.970 | R 0.971, P 0.926 |

*Production with fragment suppression. Most of the fine-tuned model's unmatched ERD
boxes looked like real animals in a visual check, so this precision is a lower bound.

On the original (Fe2O3) validation split, v1 scores P 0.963 / R 0.967 / mAP50 0.990.
The old model scores P 0.981 / R 0.970 / mAP50 0.992 there, a small regression on plates
that are not Pendifin.

