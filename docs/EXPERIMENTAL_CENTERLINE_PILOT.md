# Exp03 experimental centerline / local circular sections — pilot only

This is a standalone geometry experiment, not an application feature or a change
to existing measurements. No training, detector/segmenter inference or model
replacement is performed. Input files are only read. Reports require a new output
directory; existing reports/results are never overwritten.

## Frozen pilot

Source job `657cb33a4398`, photo `AS_MOS_Ctrl_4_004.jpg`, calibration 7.931 µm/px.
The 659 existing measurement records were inspected as metadata, not remeasured.
A contact sheet of 22 native saved masks/photo crops was visually inspected. Then
12 real masks were frozen before running the experimental geometry:

- Straight: 596, 492.
- Mild/moderate curvature: 1, 4, 321.
- Strong curvature: 0, 22, 84, 639.
- Problematic branch/fragment/edge cases: 273, 675, 663.

Categories are qualitative sample-selection descriptions, not biological ground
truth. There are 38 non-edge masks among the source's 659 saved measurements.
No multi-component masks were marked in the existing measurement metadata
(also checked across the other saved contour jobs). Therefore the gallery adds
two explicitly synthetic controls, not claimed to be real organisms:

1. A capsule plus a disconnected island (multiple-component guard).
2. A closed ring (no unique open head-to-tail path).

The old method is recomputed from each exact input bitmap and matches the saved
old numbers for all 12 real cases. Synthetic controls compare both algorithms on
the same synthetic bitmap, with the original ellipse formula calculated anew.

## Algorithm, widths and units

Implementation: `scripts/experimental_centerline.py` (pure geometry, no file I/O).
All parameters are frozen in `Config` and copied into `pilot_manifest.json`.

1. Identify all 8-connected components and retain the largest as the body
   candidate. Any original multiple-component mask is flagged regardless of size.
2. Determine reference radius as the 99th percentile of the main mask's Euclidean
   distance transform. Perform padded isotropic opening with radius
   `max(1 px, 0.35 * reference_radius)`, intersected with the original component.
   Retain its largest component. This is a scale-based appendage-separation
   hypothesis, not an anatomical classifier. Significant fragmentation or more
   than 35% removed area is flagged.
3. Show the original skeleton, skeletonize the candidate body and construct an
   8-neighbor graph with Euclidean edge lengths. Suppress diagonal shortcuts when
   an orthogonal connection exists. Iteratively remove terminal short/thin or
   very thin branches, recomputing topology after each removal.
4. Find the maximum shortest-geodesic path between remaining endpoints. Loops,
   extra endpoints, disconnected skeletons and missing endpoints are unreliable.
   A closed ring has no new length/volume, rather than an invented head/tail.
   Assign pixels to the nearest retained-path or discarded-branch skeleton seed;
   remove discarded-branch territory from the width mask as well. Thus pruning
   a branch cannot leave its full silhouette/root in the width measurement.
   Significant/ambiguous territory cuts are separately flagged and visualized.
5. Resample at about 1 px, smooth coordinates with sigma 1.5 px, extend ends to
   the first boundary of the cleaned body and resample again. Endpoint extension
   never follows an excluded raw-mask antenna. Smoothing outside the body flags
   uncertainty. Pixel skeleton geodesic and final subpixel length are separate
   fields in the result.
6. At each segment midpoint, use the segment tangent and its perpendicular.
   Trace in both normal directions to the first zero crossing of the cleaned
   body's bilinearly sampled signed-distance field (0.5 px ray step). Width is
   the sum of these two distances, not twice the distance transform radius.
   The ray stops at the first boundary; it never crosses background to include
   the opposite arm of a curled organism. Width spikes and sustained asymmetric
   cross-sections are flagged, not silently clipped to a chosen width.
7. Each interval uses its midpoint width:

   `L_px = sum(delta_l_px)`

   `V_px3 = sum(pi * (width_px / 2)**2 * delta_l_px)`

   `L_mm = L_px * (um_per_pixel / 1000)`

   `V_mm3 = V_px3 * (um_per_pixel / 1000)**3`

Circular cross-sections with thickness equal to local width are the project's
accepted biological assumption. The remaining question for review is whether
the candidate mask, endpoints and perpendicular widths identify the body
correctly. Opening may remove true thin body tips or retain thick appendage
roots; zero automatic geometric warnings is not biological confirmation.

## Reliability and review

Original crop-edge contact always remains a warning, even if opening removes
edge pixels. Discarded components, body-pixel removals, removed branch skeletons,
remaining rejected graph branches, old ellipse, new curve and width transects
are visible. All numerical outputs on flagged masks are diagnostic candidates.
No result enters aggregate statistics: `aggregate_included=false` for every
pilot case until explicit biological review. There is no automatic aggregate
mean/total length or volume and no approval-writing endpoint.

## Outputs and use

Pilot folder: `outputs/integration_exp03_local/centerline_pilot_v2/`.
The earlier v1 exploratory report is retained, not overwritten; v2 adds explicit
branch-pixel territory exclusion from the width mask after skeleton pruning.
The folder is within the isolated preview area so experiment outputs do not
invalidate the application's existing split registry. It is not app data.

- `index.html`: read-only visual comparison, layer toggles and sample filters.
- `pilot_manifest.json`: frozen sample, parameters, source/code hashes.
- `comparison.json`: all per-case numerical results and warnings.
- `cases/*/`: source crop, original mask, candidate body, removed skeleton masks,
  intermediate geometry JSON and every perpendicular measurement in CSV.
- `verification.json`: preserved input hashes/mtimes and old-result agreement.
- `review_contact_sheet.jpg`: overview of all examples and controls.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=outputs/integration_exp03_local/runtime_dependencies:. \
  .venv-ruler-ft/bin/python -B -m pytest scripts/test_experimental_centerline.py \
  -q -p no:cacheprovider

# Choose a NEW output name; existing output is deliberately refused.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=outputs/integration_exp03_local/runtime_dependencies:. \
  .venv-ruler-ft/bin/python -B scripts/compare_exp03_centerline_pilot.py \
  --output outputs/integration_exp03_local/centerline_pilot_v3

python3 -B -m http.server 8776 --bind 127.0.0.1 \
  --directory outputs/integration_exp03_local/centerline_pilot_v2
```

Analytic geometry checks cover a straight round-ended cylinder silhouette, a
known circular arc, thin leg/antenna exclusion, explicit short-branch pruning,
perpendicularity, interval integration and unit scaling, plus edge/component/
loop/empty-mask guards. These validate implementation geometry, not anatomical
accuracy on the real photographs. User visual approval is still required.
