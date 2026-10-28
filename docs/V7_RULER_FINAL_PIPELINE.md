# Final Pipeline v5 vs v7 Visual Comparison

## Ported tools and external inputs

The recovered comparison tools now live under `scripts/ruler_v7/`, rather than
inside the ignored output tree. Run from the repository root:

```bash
python3 scripts/ruler_v7/run_final_pipeline_comparison.py --check
python3 scripts/ruler_v7/run_final_pipeline_comparison.py --dry-run
```

These modes check required external inputs (including the saved duplicate
classifier) without running YOLO inference or writing comparison images. They
still require the runtime dependencies and weights. To point to alternative
locations of the same weights, use `--detector-model` and `--dedup-model`.
Defaults are `models/yolo11n_tiled_best.pt` and
`models/duplicate_pair_classifier_v7_fragment_aware.joblib`.

Running without check/dry-run performs inference and recreates its output tree
under `data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/final_pipeline_v5_vs_v7`.
The separate `run_comparison.py` reproduces the raw common-bbox comparison without
the final duplicate classifier; it accepts `--detector-model` and writes to the
historical `v5_vs_v7` output tree. Neither comparison was executed during branch
preparation. Output regeneration can replace prior generated outputs; preserve
external copies if needed.

The five comparison source IDs below are separate from the three ruler-training
source IDs according to the saved leakage audit in
`reports/ruler_v7/unseen/candidate_selection_leakage_audit.csv`.
**The six manually annotated ruler training scenes are not an independent held-out
set.** These five-image visual comparisons have no manual ground-truth metrics
and do not establish a winning model.

## Historical comparison record

Counts and parameters below are retained from the recovered report. Historical
temporary paths have had only the personal user-directory prefix removed; they
describe the old execution, not the current portable defaults above.

Leakage-free visual comparison on five full-resolution ruler/scale images. No manual ground truth labels were used and no winner is declared.

## Models
- Detector requested: `models/yolo11n_tiled_best.pt`
- Detector used: `AppData\Local\Temp\opencode\yolo11n_tiled_best.pt` (Git blob from `feature/v5-trunk-contour-model`, extracted to temp without checkout)
- v5 segmentation: `runs\segment\runs\segment\trunk_seg_v5_test20\weights\last.pt`
- v7 segmentation: `runs\segment\runs\segment\trunk_seg_v7_ruler_finetune\weights\best.pt`
- Final dedup classifier: `AppData\Local\Temp\opencode\duplicate_pair_classifier_v7_fragment_aware.joblib`

## Parameters
- tile size: 1280
- overlap: 256
- stride: 1024
- detector confidence: 0.6
- global bbox NMS IoU: 0.5
- segmentation confidence: 0.15
- segmentation imgsz: 320
- final classifier dedup threshold: 0.50
- envelope mode: disabled
- mask-IoU dedup: disabled
- restricted v5 cleanup: disabled

## Pipeline Order
1. Run tiled detector once per full-resolution image.
2. Apply global bbox NMS.
3. Send the same post-NMS bbox crops to v5 and v7 segmentation separately.
4. Apply the same v7 fragment-aware classifier dedup separately to v5 and v7 results.
5. Save final overlays, final CSVs, kept crops/masks, side-by-side full views and ruler zooms.

## Output Index
- `C_3_2_Collembola011`: 498 bbox-NMS detections, v5 final 453, v7 final 456, side-by-side `data\trunk_dataset_combined_v7\ruler_unseen_visual_comparison\final_pipeline_v5_vs_v7\C_3_2_Collembola011\side_by_side\C_3_2_Collembola011_full_side_by_side.jpg`
- `C_4_1_PS_500-1000_014`: 499 bbox-NMS detections, v5 final 469, v7 final 469, side-by-side `data\trunk_dataset_combined_v7\ruler_unseen_visual_comparison\final_pipeline_v5_vs_v7\C_4_1_PS_500-1000_014\side_by_side\C_4_1_PS_500-1000_014_full_side_by_side.jpg`
- `C_5_1_PS_500-1000_017`: 401 bbox-NMS detections, v5 final 372, v7 final 372, side-by-side `data\trunk_dataset_combined_v7\ruler_unseen_visual_comparison\final_pipeline_v5_vs_v7\C_5_1_PS_500-1000_017\side_by_side\C_5_1_PS_500-1000_017_full_side_by_side.jpg`
- `C_1_2_Collembola005`: 1288 bbox-NMS detections, v5 final 1240, v7 final 1238, side-by-side `data\trunk_dataset_combined_v7\ruler_unseen_visual_comparison\final_pipeline_v5_vs_v7\C_1_2_Collembola005\side_by_side\C_1_2_Collembola005_full_side_by_side.jpg`
- `C_1_3_Collembola006`: 1180 bbox-NMS detections, v5 final 1126, v7 final 1123, side-by-side `data\trunk_dataset_combined_v7\ruler_unseen_visual_comparison\final_pipeline_v5_vs_v7\C_1_3_Collembola006\side_by_side\C_1_3_Collembola006_full_side_by_side.jpg`
