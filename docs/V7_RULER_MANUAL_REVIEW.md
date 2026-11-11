# Raw Contour Review

## Remote SSH: headless web review of the 57 BOTH_BAD cases

The Linux remote has no `DISPLAY`/`WAYLAND_DISPLAY` and its active Python has no
`cv2`. The Tkinter tools below therefore cannot be used there as-is. Use the
headless review instead; it needs only system Python and Pillow, no model runtime:

```bash
python3 -B scripts/ruler_v7/web_ruler_error_review.py --check
python3 -B scripts/ruler_v7/web_ruler_error_review.py --port 8765
```

In VS Code **Ports**, forward remote port **8765** and open the displayed local
address in a browser (normally `http://localhost:8765`). The server binds only to
remote `127.0.0.1`. Keep that terminal running; Ctrl+C stops it. Repeating the
same command resumes the first incomplete case from saved decisions.

The default asset root is the verified external snapshot:
`/home/jana/collembola-rescue/incoming/ruler-v7-20261027T162453Z/snapshot/repo/`.
Use `--assets-root` if its location changes. It reads the existing raw crop and
V5/V7 masks directly; it does not copy assets, run inference, or create new
comparison/contact-sheet files. Browser canvases render the stored masks as
contours, a filled overlay, or a binary mask. Wider ruler context is cropped from
the source photograph on demand in memory; its red box matches the recorded
floor/ceil crop bounds. Click a panel for a larger view.

### What is finished, and what needs this review

- Position review: 130/130, including 93 R cases, finished.
- V5/V7 relative ratings: 93/93, including 57 BOTH_BAD, finished.
- Earlier coarse cause review: 15/57 finished; 42 lacked a cause.
- New detailed review: a framing decision plus one error category **per model**.
  The old 15 causes are read-only references, not automatically inferred V5/V7
  categories. The 42 cases without an earlier cause appear first, followed by
  the 15 with earlier references. All 57 need the new per-model fields.

For each case, select whether the body is complete in the crop, cut by its edge,
or unclear. Then rate V5 and V7 separately: BODY_PART_MISSING, RULER_INCLUDED,
TOO_LARGE, TOO_SMALL, MULTIPLE_PROBLEMS, or UNASSESSABLE. Every click commits
immediately. The next incomplete case opens once its three choices are saved.
Missing masks have an explicit badge and are never replaced with invented masks.

### Independent storage and summary

New decisions are saved only under ignored
`outputs/ruler_v7_body_error_review/`:

- `review.sqlite3`: append-only event history; revising a new answer appends a
  revision without deleting an earlier answer/timestamp.
- `summary.json`: progress, crop/framing evidence and a provisional next-step
  recommendation only when the detailed review is complete.
- `summary_by_source_model_error.csv`: counts by source, model, framing condition
  and error type, including `ALL` aggregate rows and explicit `PENDING` entries.

`--output-dir` chooses a separate review directory. Input table fingerprints
prevent accidentally resuming decisions against a changed catalog. Original
position/relative-rating/cause CSVs, source/detection IDs, masks, annotations and
weights remain untouched. An answer is considered saved only after the browser
receives confirmation; failed requests do not automatically advance.

The final next-step evidence distinguishes truncated input bodies (try crop /
padding first) from segmentation errors on complete bodies or inclusion of ruler
marks (new targeted mask annotations, then consider fine-tuning). Mixed evidence
supports a combination. An unclear review is not sufficient justification for a
dataset/model change. No training or production-model change is triggered by this
tool. **The six training ruler scenes are not an independent held-out test.**

## Independent reference annotation after the completed error review

Do not reopen the finished 57-case error rating to draw reference polygons. The
separate `prepare_reference_annotation_pack.py` tool builds the explicitly listed
12-case pack from `outputs/ruler_v7_body_error_review/annotation_candidates.csv`;
it refuses to overwrite an existing pack. Its output is
`outputs/ruler_v7_reference12_v1/`, containing unchanged raw crops, wider clean
annotation images and context, separate V5/V7 reference masks, hashes and source
coordinate mappings. It creates no initial annotation JSON and assigns no data
to training. `README.md` in the pack describes the exact 3/3/2/2/2 selection and
the source-level split proposal (not applied).

```bash
python3 -B scripts/ruler_v7/web_reference_annotation.py --port 8766
```

Forward port 8766 in VS Code. Draw the target body from scratch: click vertices,
Enter closes, Z undoes, drag vertices to correct, wheel zooms, Space+drag pans,
S saves a new LabelMe polygon and proceeds. Mask references are separate and
collapsed by default; there is no prediction-to-polygon import. New manual
LabelMe JSON files go to the pack's `annotations/`, and every save is also kept
in `annotation_history/`. The larger annotation images may contain other bodies;
this is a single-target reference task, not a fully annotated training dataset.

Potential future split: C_1_2_Collembola005 and C_1_3_Collembola006 for training,
C_3_2_Collembola011 for validation, and C_4_1_PS_500-1000_014 plus
C_5_1_PS_500-1000_017 reserved from all training/tuning. The latter two were
already visually inspected; they are not a blind untouched test. Confirm source
membership and final evaluation protocol before any dataset/model change.

## Additional training-source annotations (awaiting approval to launch)

`prepare_additional_training_annotations.py` prepares a separate 22-case pack at
`outputs/ruler_v7_additional_train22_v1/`: 11 targets from C_1_2_Collembola005
and 11 from C_1_3_Collembola006. It excludes all prior 12 IDs, truncated and
unassessable cases, non-R targets, and every validation/final-test source.
`selection.csv` gives exact original asset paths and saved V5/V7 selection reasons.
The original raw PNG is the annotation image (byte-identical, native dimensions),
not a padded image or comparison panel. Context and prediction references are
separate. No YOLO dataset is built.

The shared web editor now accepts arbitrary nonempty, uniquely keyed packs.
The additional pack opts into automatic save-and-next on closing a valid polygon;
closed-vertex edits save on release. Legacy packs keep explicit-save behaviour.
V5/V7 contours or binary masks can be enabled separately; they are never copied
into a manual polygon. Saved LabelMe files preserve raw dimensions, source crop
origin and IDs; immutable revisions and resume are supported.

User confirmation of the 22-case selection is required **before** launching a
server for this pack. The proposed port is 8767. Existing 12-case files must not
be overwritten, and validation/held-out sources must not be added to this pack.

## Preparing the isolated manual-label dataset candidate (no training)

`build_reviewed_ruler_dataset.py` reads the 12+22 manual annotations, maps their
unchanged vertices into full-source coordinates, and prepares standard/tight
native-resolution candidates under `outputs/ruler_yolo_seg_candidate_v1/`.
It refuses to overwrite an existing output. Each of the 34 references requires
an explicit visual decision in `audit_decisions.json` before export. Uncertain
neighbour cases are blocked, not treated as background. A paired candidate for
det713/det722 uses both existing manual polygons; no model mask becomes a label.

The separate `audit_reviewed_ruler_dataset.py` checks actual exported image pixels
against the source photographs, reparses YOLO labels, verifies vertex roundtrip
and source separation, then writes a read-only `index.html` gallery. Current
candidate results are 20 train images/21 polygons, 1 val image/1 polygon and
3 test images/3 polygons. Nine references need neighbour/boundary work; one
reference is covered by the two-body paired image instead of duplicating it.
Existing source annotations and datasets are not overwritten. No training is
started by either script. The output README and manifests document all reasons
and the small-sample/non-blind-test limitations; user visual approval is pending.

## Portable legacy review commands

Tools are under `scripts/ruler_v7/`; preserved review tables are under
`reports/ruler_v7/manual_review/`. Original crop/mask paths in `index.csv` still
refer to external `data/` artifacts. Restore those images and masks before
opening the GUI or running input checks. No images, masks or model weights are
part of this branch's new file set. Font loading tries Windows Arial, Linux
DejaVu/Liberation, then Pillow's default.

```bash
python3 scripts/ruler_v7/manual_contour_review.py --check
python3 scripts/ruler_v7/review_both_bad.py --check
```

These commands check inputs without GUI/CSV writes and do not run models. Run
without `--check` to continue the interactive review; the GUI updates its review
CSVs when decisions are made or saved. `manual_contour_review.py --summary`
rewrites its summary CSV from existing decisions. These write modes were not
run during branch preparation.

`build_review.py --inspect` checks the external raw-mask inventory without
writing. Running it without `--inspect` regenerates comparisons/indexes in the
ignored historical `data/.../raw_contour_review_v5_vs_v7/` output directory. It
does not overwrite the preserved report copies under `reports/`. Do not replace
those copies without comparing IDs and existing decisions.

## Scope of the saved decisions

The raw ruler index has 779 detections. Of 130 prioritized disagreement cases,
the position review classified 93 as R (body contacting a ruler mark), 3 as N,
9 as X and 25 as U. The 93 R cases were rated: 2 V5_BETTER, 34 V7_BETTER and
57 BOTH_BAD. The reason review covers a source-balanced sample of 15 of the 57
BOTH_BAD cases: 8 BODY_PART_MISSING and 7 MULTIPLE_PROBLEMS. Source/detection IDs,
notes, decisions, timestamps and saved counts are preserved from the snapshot.

This deliberately prioritized review is not an unbiased estimate over all 779
detections. **The six manually annotated ruler training scenes are not an
independent held-out set**; all three of their source IDs contributed to v7
training. The five unseen comparison photographs are a separate visual review,
not a ground-truth detection-recall evaluation. Missing masks for an existing
bbox are distinct from organisms that the upstream detector missed entirely.

## Historical rendering record and external review sheets

The historical sheet paths below remain references to external generated images.

Source: saved all_crops and all_masks before dedup. No models run.
All bbox IDs are included regardless of later retention. Blank mask paths mean no saved raw mask.
Same crop, scale, contour colour and thickness in every comparison. No metrics on image panels.
Difference sheets prioritize missing-mask and large mask disagreement cases, not a winning model.
ROI selection includes ticks, digits and nearby bodies; see ruler_region_overview.jpg and summary.json.

## Review Order
1. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_001.jpg`
2. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_002.jpg`
3. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_003.jpg`
4. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_004.jpg`
5. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_005.jpg`
6. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_006.jpg`
7. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_007.jpg`
8. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_008.jpg`
9. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_009.jpg`
10. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_010.jpg`
11. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/01_differences_011.jpg`
12. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_001.jpg`
13. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_002.jpg`
14. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_003.jpg`
15. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_004.jpg`
16. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_005.jpg`
17. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_006.jpg`
18. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_007.jpg`
19. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_008.jpg`
20. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_009.jpg`
21. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_010.jpg`
22. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_011.jpg`
23. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_012.jpg`
24. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_013.jpg`
25. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_014.jpg`
26. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_015.jpg`
27. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_016.jpg`
28. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_017.jpg`
29. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_018.jpg`
30. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_019.jpg`
31. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_020.jpg`
32. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_021.jpg`
33. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_022.jpg`
34. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_023.jpg`
35. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_024.jpg`
36. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_025.jpg`
37. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_026.jpg`
38. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_027.jpg`
39. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_028.jpg`
40. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_029.jpg`
41. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_030.jpg`
42. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_031.jpg`
43. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_032.jpg`
44. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_033.jpg`
45. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_034.jpg`
46. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_035.jpg`
47. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_036.jpg`
48. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_037.jpg`
49. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_038.jpg`
50. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_039.jpg`
51. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_040.jpg`
52. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_041.jpg`
53. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_042.jpg`
54. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_043.jpg`
55. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_044.jpg`
56. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_045.jpg`
57. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_046.jpg`
58. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_047.jpg`
59. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_048.jpg`
60. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_049.jpg`
61. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_050.jpg`
62. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_051.jpg`
63. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_052.jpg`
64. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_053.jpg`
65. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_054.jpg`
66. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_055.jpg`
67. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_056.jpg`
68. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_057.jpg`
69. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_058.jpg`
70. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_059.jpg`
71. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_060.jpg`
72. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_061.jpg`
73. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_062.jpg`
74. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_063.jpg`
75. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_064.jpg`
76. `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/raw_contour_review_v5_vs_v7/02_all_ruler_065.jpg`
