# Application measurement: curved centerline and integrated circular sections

The application now uses **only** `centerline_circular_sections_v2` for length
and volume. The approved algorithm in `scripts/experimental_centerline.py` is
reused without changing its pilot parameters. There is no application selection
for equivalent-ellipse/SAM-axis measurement. Offline historical tools and frozen
pilot comparisons remain archived; their old numbers are not presented as current
application results.

## Definitions

- Length: arc length along the cleaned curved body centerline.
- Width measurements: first body-boundary intersection in each perpendicular
  direction at segment midpoints; discarded-branch territories are removed from
  the measurement mask, not merely from the skeleton.
- The table's single descriptive width is the **median local width**. It is not
  used as a constant-width cylinder in the volume calculation.
- Volume: `sum(pi * (w_i / 2)^2 * delta_l_i)` over all valid local sections.
- Pixels to mm: `L_mm = L_px * calibration_um_per_px / 1000`.
- Volume: `V_mm3 = V_px3 * (calibration_um_per_px / 1000)^3`.
- Area/perimeter/centroid now describe the extracted body. `mask_area_px` also
  retains the source mask's foreground count for provenance.

## Reliability policy

The algorithm flags raw crop-edge contact, multiple components, extraction
failure, loops/remaining forks, uncertain endpoints, off-body paths, width spikes
or asymmetry and ambiguous branch-territory cuts. Such rows are orange and say
`nepouzdano`, with `volume_included=false`. Diagnostic candidate values may remain
visible, but **none** enter total, mean, median, minimum or maximum volume.

The summary shows candidate, reliable and excluded counts. All statistics use
only reliable, calibrated rows. No calibration means physical values are null.
A calibrated image with no eligible rows has an included total of zero and a
count of zero; this does not assert that the excluded organisms have zero volume.

## Application paths

- `backend/services/centerline_measurement.py`: approved geometry adapter,
  reliability fields and reliable-only aggregation.
- `backend/services/contours.py`: unchanged exp03 segmentation/dedup followed by
  the new measurement, CSV/Excel and curved-path/body overview.
- Single-image measurement compatibility API, project measurement and project
  processing all delegate to this same workflow. Older API method-name strings
  cannot select an archived measurement algorithm.
- Project processing does not overwrite human annotations with auto-accepted
  boxes. Saved reviewed boxes are snapshotted into the job. Rejected boxes are
  omitted, added/corrected boxes participate in segmentation, and annotation IDs
  are preserved in provenance.

## Reusing masks and handling prior results

`GET /api/contours/source/{image_id}` discovers saved same-image exp03 masks.
The UI prefers the latest source and can remeasure without detector or segmenter
inference. New output always gets a new job directory; old result files are never
overwritten. A saved old-method result is hidden until remeasurement; its numeric
data and export are not shown by the new UI. Project ZIP export requires current
method results to prevent mixing methods.

Source masks can only be reused for the same checkpoint/image. If saved human
annotations changed, a canonical annotation hash invalidates stale crop reuse.
New segmentation then uses only the non-rejected reviewed boxes, without running
the detector again. Unchanged annotations and calibration-only changes reuse
the stored masks. Measurement never writes the human annotation JSON.

## Presentation and exports

- Table: reliability and inclusion appear beside Detection ID; excluded numeric
  volumes have diagnostic styling and a tooltip.
- Selected organism: body contour, curved line, local perpendicular sections and
  discarded branches; no equivalent ellipse.
- **Contours** overview: ordinary yellow external contours from the saved
  segmentation masks, over the original photo. It has no centerline, local
  transects or discarded-branch overlay, including after closing View Results.
- **View Results** right-hand detail: only after selecting an organism, the
  original photo plus that organism's body contour, centerline, local widths
  and discarded branches. Detail geometry is never placed on the main viewer.
- CSV and Excel: method version, reliability/inclusion, warnings and new metrics.
- `centerline_local_widths.csv` and Excel **Local widths** sheet: each local width,
  interval length and volume contribution, including inclusion flags.
- Excel **Method** sheet: exact integration formula and aggregation policy.

The local immutable build is served with the API on port 8774. No training,
model-weight replacement, commit or push is part of this integration.

The authenticated `GET /api/contours/overview/{job_id}` renders the plain external
outline view in memory. It reads the original photo and existing masks with
`RETR_EXTERNAL`, then returns a display-only JPEG. It does not recalculate any
measurement or modify the source photo, masks, geometry JSON or stored overlays.
