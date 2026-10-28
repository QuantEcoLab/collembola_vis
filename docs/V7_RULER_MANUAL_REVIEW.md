# Raw Contour Review

## Portable review commands

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
