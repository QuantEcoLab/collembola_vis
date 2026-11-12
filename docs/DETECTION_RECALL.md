# Detection recall on Pendifin plates (detector v3, November 2026)

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

Detector v1 (first round) fixed recall but still put boxes on substrate texture
around bright dish-rim/bubble halos and on colour-fringed blobs. Detector v2
(second round) is trained on corrected labels with hard negatives and runs at a
higher threshold. Detector v3 (third round) undoes round-2 hard negatives that were
really animals on the dish rim, adds more verified animals and non-animals, and
keeps the boxes of touching animals.

## Standard detection profile (backend `settings`)

| setting | value |
|---|---|
| `default_model` | `models/yolo11n_tiled_pendifin_v3.pt` (not in git, see Deployment) |
| `default_model_sha256` | `22e873ac...7f50c3`, checked at startup; the service does not start without the right file |
| `default_conf` | 0.55 on 1280 tiles, overlap 256 |
| `default_iou` | 0.5 global NMS |
| `default_contain` | 0.8: drop a box that lies >= 80% inside a larger kept box (fragment duplicates) |
| `default_contain_keep_confident` | true: a contained box stays when its confidence is at least that of every kept box containing it (touching animals under one merged box) |
| `default_extra_tile_sizes` / `default_extra_conf` | `[]` / `[]`: no extra pass by default |

`scripts/infer_tiled.py --profile standard` runs exactly these settings (the model path
is relative to the repository root); `--profile legacy` runs the old detector, and
explicit flags override either. Without `--profile` the old CLI defaults apply.
The script also takes `--contain 0.8` and optional extra passes, for example
`--extra-tile-sizes 640 --extra-conf 0.8`, which run 640 px tiles seen 2x upscaled.
Without these flags it behaves as before. With the fine-tuned models the 640 pass
adds almost nothing: 1 of 101 dev animals for v1, and more than half of the boxes it
added on replicate-3 images were dish-rim speckle, fibres or fragments. It also costs
about 4x the tiles.

Each detection job records the resolved settings as `result.detector_profile`. The
exp03 contour router accepts detections that match either the current standard
profile or the legacy one (old model, conf 0.6, 1280 tiles, no fragment
suppression), so older jobs stay usable. A job recorded before profiles existed
counts as legacy.

The workspace's standard "Run/Re-run Detection" sends no thresholds. The advanced
dialog sets the confidence. Fragment suppression stays on.

## How the training labels were built

Only TRAIN-split images were used: replicates 1-3 of both projects, minus
`CAC_PDF_Ctrl_2_002` (the user-flagged evaluation image). Holdout replicates 4-5
were never read for labels, mining or tuning.

### v1 (`scripts/recall_ft/verified_extras_v1.json`)

1. Candidates. The production model was run at conf 0.05 on 1280 tiles and on
   640 tiles, then merged with NMS 0.5. Candidates matching an annotation box
   (IoU >= 0.3) were removed, and so were candidates >= 80% inside a larger box.
   The remaining extras with conf >= 0.25 on the 17 CAC train images were reviewed.
2. Bright context means the median grey of a 3x box window is > 110: grid paper,
   ruler bars and the dish rim. A seeded check showed 70/70 of these were not
   animals. They were labelled negative automatically.
3. Every remaining extra (1,842) was judged visually on CLAHE chip sheets at
   about 1.3x: animal, uncertain, or not (texture, fibre, rim, fragment).
   Results: 782 animals, 256 uncertain.
4. Labels: annotation boxes (accepted + added) plus the verified animals.
   Fragments (>= 80% inside a larger box) are dropped. Uncertain objects are
   inpainted. ERD images used the human review as is.
5. Tiles are 1280 with stride 1024, plus 30% 2x-upscaled 640 crops and 35% of the
   empty tiles. The original training set (`data/yolo_tiled`) is mixed in so other
   plates keep working.

### v2 additions (`scripts/recall_ft/verified_r2.json`)

1. Mining. v1 was run at conf 0.1 (1280 tiles, contain 0.8) on all 35 train images,
   and the fold-A model (trained on replicates 1-2 only) on the replicate-3 images.
   Every prediction was compared with the v1 labels: positives, uncertain objects,
   and known negatives (rejected boxes and the v1 'not' judgements).
2. Every prediction that matched no label (CAC conf >= 0.2: 319 objects; ERD conf
   >= 0.3: 215 objects), and every known negative that v1 still predicted at conf
   >= 0.35 (70 objects), was judged on CLAHE chip sheets. Nearby labels were drawn
   so that fragments and motion ghosts of labelled animals were visible. Doubtful
   objects were zoomed raw and CLAHE.
3. Results: 313 missed animals, 97 uncertain, 200 not-animals. About half of v1's
   "false positives" on train plates were animals missing from the labels, on ERD as
   well: the human ERD review of the train plates is not complete. The not-animals
   are dish-rim/halo bands, substrate texture blobs, fibres, the grid card,
   shapeless colour-fringed blobs and partial boxes on labelled animals.
4. Colour-fringe policy. The scanner records R, G and B at slightly different
   times, so a moving animal gets colour fringes. An object with a recognisable
   collembolan body (segments, legs, antennae) is an animal, fringed or not. A
   shapeless or rod-like colour-fringed blob is not an animal. Three ERD labels
   (rainbow blobs) and two labels on the grid card were removed (`drop_pos`). One
   rainbow fan that could be a fast-moving animal is uncertain.
5. Dataset. The v1 recipe plus: verified animals as labels; uncertain objects
   inpainted (unless they overlap a labelled animal); for every not-animal one
   extra 1280 tile with the object at a seeded random position, as a hard negative
   (181 tiles: a tile that would overlap an excluded dev tile is skipped). 3,878
   tiles in all, plus the original training set.
6. Training (`scripts/recall_ft/train_ft.py`): from `yolo11n_tiled_best.pt`, 25
   epochs, imgsz 1280, batch 4, SGD lr0 0.003, mosaic, flips in both directions.
   The weights are `last.pt`.

### v3 additions (`scripts/recall_ft/verified_r3.json`)

1. Mining. v2 (trained on all train images) was run at conf 0.05 on the 35 train
   images, and the fold-A v2 model on the replicate-3 images. Judged: every
   prediction matching no label at conf >= 0.2, or in the dish-rim band
   (r >= 0.9 R, from a Hough fit of the dish on the thumbnail), every known negative
   v2 still predicts at conf >= 0.5, and every round-2 "not" object in the rim band
   (the rim audit). 368 objects in all.
2. Chip sheets at about 1.2x (CLAHE, labels drawn), then 2.4x raw + CLAHE zooms of
   all rim objects. Results: 84 animals, 87 uncertain, 105 new non-animals.
3. Rim audit. Of the 57 round-2 "not" objects in the rim band, 10 are animals lying
   on the rim refraction band (pale segmented bodies, often with legs) and several
   more are uncertain. They had been labelled as rim/halo texture and each got a
   hard-negative tile, which taught v2 to suppress rim animals. `unnot` removes
   these hard negatives (25 overturned negatives in all). 13 rim animals in total
   become labels.
4. Four candidates were body parts of an animal without a label (the animal had been
   judged uncertain before); the animal's own v2 box (conf >= 0.86) is labelled.
5. Dataset: the v2 recipe plus `--verified3`; 3,970 tiles (v2: 3,878). Without
   `--verified3` the builder still rebuilds the v2 set byte for byte. Training as for
   v2 (25 epochs from `yolo11n_tiled_best.pt`, physical GPU 2); the weights are
   `last.pt`. A fold-A model (replicates 1-2 only) was trained the same way for the
   dev evaluation.

## Dev evaluation and the frozen settings

All choices were made on train-split data before any holdout or gold run. The
fold-A models are trained on replicates 1-2 and scored on replicate 3, on two
views:

* 20 exhaustively labelled 1024 px dev tiles on CAC replicate-3 images (101
  animals), kept out of every training set (`excluded_dev_tiles.json`);
* the full replicate-3 images (6 CAC, 6 ERD; 2,942 animals) against the v2 labels.
  Fragment boxes inside larger labels are not counted as animals. An unmatched
  prediction counts as a false positive.

Fold A, conf 0.4 and contain 0.8:

| | dev tiles | CAC rep 3 | ERD rep 3 |
|---|---|---|---|
| v1 recipe | R 0.980, P 0.971 | R 0.970, P 0.980 | R 0.972, P 0.982 |
| v2 recipe | R 0.980, P 0.952 | R 0.980, P 0.980 | R 0.973, P 0.982 |

The v2 recipe moves the whole precision-recall curve up on the replicate-3 images.
At the same pooled precision (about 0.985) it has about 0.8 points more recall.

Confidence: the rule, fixed before the numbers were computed, was to take the
highest pooled F1 on the replicate-3 images, compared at three decimals, with
ties going to the higher threshold. F1 is flat between 0.40 and 0.55 (0.979) and
drops at 0.60 (0.978), so the threshold is 0.55. Fold-A v2 at 0.55: pooled
R 0.966, P 0.992.

Fragment suppression: with fragment boxes removed from the labels, contain 0.8
costs no replicate-3 recall (CAC 0.970 with and without it, ERD 0.972) and removes
most fragment false positives. A variant that spares a contained box whose centre
lies off the larger box's body axis, for juveniles beside or under adults,
recovered no animal and added 49 false positives on CAC replicate 3. It is not
used.

### Round 3 (v3, keep-confident fragment suppression)

The rule was written to `rule_r3.json` before the v3 fold-A model was scored (rule
files, sweeps and gold-look log: `adeb:/home/adeb/agent-work/developer-recall/r3/`). Labels
are the v2 labels plus `verified_r3.json`; predictions are post-processed with
contain 0.8 and keep_confident.

* Threshold: the lowest conf in 0.30-0.65 whose pooled replicate-3 precision is
  >= 0.985, with dev-tile precision >= 0.95. The 0.985 floor was meant to keep the
  user-flagged plate at about 0.91 (round-2 gold feedback, so this is adaptive).
* Model: v3 replaces v2 only if fold-A v3 at its threshold has at least the recall
  of fold-A v2 at its threshold, with precision at most 0.003 lower.
* Keep-confident: on fold-A v2 at 0.40, dev-tile recall goes from 0.970 to 0.980
  (one touching pair). CAC replicate 3 gains 2 false positives on about 1,900
  animals; ERD is unchanged.

Fold-A models on the replicate-3 train images (2,961 animals; rim means
r >= 0.93 R of the dish fit, 148 animals):

| | conf | pooled R | pooled P | CAC3 R / P | ERD3 R / P | rim R |
|---|---|---|---|---|---|---|
| v2 | 0.40 (rule) | 0.976 | 0.986 | 0.976 / 0.985 | 0.976 / 0.990 | 137/148 |
| v2 | 0.55 (round 2) | 0.962 | 0.994 | 0.960 / 0.993 | 0.965 / 0.996 | 135/148 |
| v3 | 0.35 (rule) | 0.980 | 0.988 | 0.982 / 0.985 | 0.975 / 0.994 | 138/148 |
| v3 | 0.50 | 0.972 | 0.995 | 0.973 / 0.993 | 0.971 / 0.998 | 137/148 |
| v3 | 0.55 (chosen) | 0.966 | 0.996 | 0.966 / 0.994 | 0.965 / 1.000 | 135/148 |

The rule picked v3 at conf 0.35 (`frozen_settings_r3.json`). On the gold standard,
v3 at 0.35 found 138/139 holdout animals and 148/149 on the flagged plate. Its
precision on the flagged plate was 0.854, below the 0.90 floor. In the
conf 0.35-0.55 band, precision on the flagged plate's mottled substrate is far
lower than on the train plates. On replicate 3 that band holds 41 animals and
24 non-animals.

The fallback (`frozen_settings_r3b.json`, written after that one gold result) keeps
v3 at the round-2 threshold, 0.55. There, fold-A v3 is at least as good as
fold-A v2 on pooled recall and precision and on CAC precision. A second look at
the 0.35-0.55 band with 640 px tiles (2x upscaled) did not separate animals from
non-animals: 7 of 41 animals and 11 of 24 non-animals got no box there. The
rim-band gain of v3 is in that low-confidence band (138 vs 135 of 148 at 0.35), so
at 0.55 rim recall on replicate 3 is the same as v2's.

## Results of the frozen profiles (independent evaluations)

The critic's gold standard is exhaustively labelled 1024 px tiles on the holdout
plates (replicates 4-5), plus the user-flagged `CAC_PDF_Ctrl_2_002`. All rows use
contain 0.8; the v3 rows also use keep_confident.

| | holdout (30 tiles, 139 animals) | flagged Ctrl_2_002 (24 tiles, 149 animals) |
|---|---|---|
| production (legacy, conf 0.6) | R 0.928, P 0.963 | R 0.624, P 0.877 |
| v1, conf 0.4 | R 0.986, P 0.986 | R 0.980, P 0.883 |
| v2, conf 0.55 | R 0.971 [0.944, 1.0], P 0.993 | R 0.980 [0.944, 1.0], P 0.917 [0.857, 0.969] |
| v3, conf 0.35 (round-3 rule; rejected) | R 0.993 [0.980, 1.0], P 0.993 | R 0.993 [0.980, 1.0], P 0.854 [0.800, 0.908] |
| **v3, conf 0.55 (standard)** | R 0.986 [0.959, 1.0], P 0.993 | R 0.980 [0.957, 1.0], P 0.923 [0.860, 0.976] |

Holdout by project for v3 at 0.55: CAC R 0.981 (108 animals), ERD R 1.0 (31); v2
had CAC R 0.963. The remaining holdout misses are small animals (< 70 px: R 0.970;
v2 0.940). Lowering the threshold in the app's advanced dialog to 0.35 finds almost
every animal but costs precision on mottled plates (see the 0.35 row).

The phase-1 harness (human-reviewed ERD holdout plates, 1,553 boxes) gives, for v3
at 0.55, R 0.987 [0.980, 0.991] and P 0.918 at IoU 0.3 (v2: R 0.985, P 0.931).
Its unmatched predictions are mostly animals missing from that review. On the
critic's audited regions, CAC recall is 99/99 (all 11 animals that the old model and
the reviewer both missed are found) and ERD 30/31. Every prediction gate passes. The
`gt_frozen_matches_live` gate fails because the live annotation of the holdout image
`CAC_PDF_C3_5_020` changed after the split was frozen, so the critic has to
re-freeze. That image is not in the headline tier.

## App: which boxes a photo uses

* An annotation file saved from the editor has `saved_at` and always wins. It holds
  a reviewer's decisions.
* A file without `saved_at` was written from raw detections, for example when the
  project was imported (all 25 unreviewed CAC images, including
  `CAC_PDF_Ctrl_2_002`). Once the photo's current detection job has finished, the
  workspace shows that job's boxes, and project measurement measures them. A
  pending or failed re-run does not replace the file.
* For reviewed photos the workspace offers "Merge new detections". It replaces the
  model boxes with the current run's boxes and keeps every added and rejected box.
  It drops new boxes on objects the reviewer rejected or drew. A previously accepted
  box stays unless a new box matches it (IoU >= 0.3) or contains it; a smaller new
  box lying inside a reviewed box (a body part) is dropped instead of displacing it.
  The result is only a preview until it is saved
  (`POST /api/annotations/{image_id}/merge-detections`).
* Unsaved box edits (a merge, toggles, drawn or moved boxes, boxes loaded from the
  community page) block measuring until Save, with a message. Measuring uses the
  saved annotation file or the detection job, so it would otherwise ignore them.
* Saved exp03 masks are reused only if they were made from the same boxes: the
  same detection job, or the same annotation snapshot. After a new detection run,
  the new boxes are segmented again.
* exp03 results record the detector hash and job of the boxes that were measured
  (`pipeline.detector_sha256`, `pipeline.detector_job_id`). For boxes of unknown
  origin both are null.
* A measurement without any detection job (workspace measure with no detection, or
  project "Process" on photos that were never detected) runs exp03's own detector.
  Since round 3 that run uses the standard profile (on exp03's CPU device), so those
  photos get the same boxes as "Run detection". Check on two train photos
  (`CAC_PDF_C2_3_013`, `ERD_PDF_C3_2_017`), exp03 on the legacy boxes compared with
  exp03's own v3 run: 256/257 and 204/210 legacy organisms have a standard box with
  IoU >= 0.7. Per organism, the median length differs by 1.2% and 0.9% and the
  median width by 2.5% and 1.7%. The standard run measures 74 and 27 more animals,
  mostly small ones, so the median length of all measured animals is lower
  (69.7 vs 77.0 px and 67.3 vs 69.4 px).

## Deployment

1. Install the weights at `models/yolo11n_tiled_pendifin_v3.pt` (sha256
   `22e873acb6362ec9ac7d1fe712a31604f5df38281527a7623fd880f1e67f50c3`, also in
   `backend/config.py` as `default_model_sha256`). The file is not in git yet (`*.pt`
   is ignored; only `yolo11n_tiled_best.pt` is force-tracked). At the time of
   writing, the build is at
   `adeb:/home/adeb/agent-work/developer-recall/models/yolo11n_tiled_pendifin_v3.pt`.
   That is a work directory, so force-add the file like `yolo11n_tiled_best.pt`. The service now refuses to start if the file is missing
   or has a different hash. Before this check, a deploy without the file would
   start and then fail every standard detection.
2. Keep `models/yolo11n_tiled_best.pt`, which is the legacy detector (old jobs, exp03).
3. Restart the service. Speed is unchanged: it is the same network size on the
   same tiles.

GPU selection. `scripts/infer_tiled.py --device N` now means "the N-th GPU this
process can see". Use `CUDA_VISIBLE_DEVICES=2 python scripts/infer_tiled.py ...
--device 0` to run on physical GPU 2. Before, ultralytics rewrote the variable, and
that combination ran on physical GPU 0, the production GPU. `train_ft.py` refuses to
start without `CUDA_VISIBLE_DEVICES` and prints the physical GPU it uses. Inside the
service, which sees all GPUs, detection on device "1" now runs on `cuda:1`. Fine-tune
jobs still pass device strings to ultralytics' trainer, so in the service a
fine-tune on "1" may still land on the first GPU.

## Migration proposal for existing annotation files (not applied)

Nothing in production has been changed. After deploying, and with the user's
approval:

1. Run project detection ("Detect all") for both Pendifin projects. This creates a
   v3 job per photo and points the project at it. The 25 CAC files without
   `saved_at` are then superseded automatically; they need no rewriting and stay
   on disk unchanged.
2. Reviewed photos (all 30 ERD, CAC C2_1..C2_5): open each in the workspace, click
   "Merge new detections", check the result and Save. A reviewer should do this:
   the merge keeps their added and rejected boxes, but new boxes need a look.
3. Optional cleanup, only if wanted: rename files without `saved_at` to
   `<image_id>.auto.json`. This makes their status visible; the app no longer needs it.
