# Controlled exp02_os5 ruler review

Open **http://127.0.0.1:8770/**. The interface starts in TRAIN mode.

## Frozen review bundle

`outputs/ruler_exp02_os5_manual_validation_v1/` contains the exact 779-case
inventory, source and crop hashes, configuration seals, and frozen predictions.

| Role | Source photograph | Cases |
| --- | --- | ---: |
| train | C_1_2_Collembola005 | 236 |
| train | C_1_3_Collembola006 | 331 |
| val | C_3_2_Collembola011 | 85 |
| test | C_4_1_PS_500-1000_014 | 63 |
| test | C_5_1_PS_500-1000_017 | 64 |

Fixed checkpoint SHA-256:
`9540b6478a8d77fcf6f2aa7db5c639c812e3c91953baa57c58670903221fc958`.
Inference completed for all 779 crops: 772 have a candidate mask and seven have
an empty mask. Initial V7 masks are available for 751 crops. The original raw
crop defines the annotation coordinate system.

## Decisions

| Key | Saved category | Meaning |
| --- | --- | --- |
| 1 | ACCEPTED_UNCHANGED | Accept the candidate outline, TRAIN only |
| 2 | MANUALLY_CORRECTED | Save an edited TRAIN outline, or confirm manually drawn/reference GT in VAL/TEST |
| 3 | FALSE_DETECTION | No reliable target organism |
| 4 | AMBIGUOUS_OR_TRUNCATED | Unclear or incomplete target |
| 5 | UNLABELED_NEIGHBOUR | A neighbouring organism prevents a reliable complete annotation |
| B | — | Previous case |
| Left arrow | — | Previous case |
| N / Right arrow | — | Next case |
| U | — | Return the most recent active saved decision to PENDING; preserve history |

**Brzi pregled** (Fast review) is enabled by default and remembered in this
browser. One press of 1–5 shows a brief visual decision banner, saves, and loads
the next pending case within the same role, wrapping to earlier pending cases
when needed. For decisions 1 and 2, the press explicitly confirms both whole-body
and no-unlabelled-neighbour checks: both are stored as `true`, with
`confirmation_source=FAST_REVIEW_EXPLICIT_DECISION`. Decision 2 saves the current
edited outline. With fast review disabled, these two checkboxes must be checked
manually. Decision 1 remains TRAIN-only in both modes.

Only positive, geometrically valid, explicitly decided TRAIN
records satisfying these checks receive `training_candidate=true`. A draft is
always pending. Categories 3–5 do not become training candidates or background
examples.

Holding a shortcut does not repeat saves: repeated keydown events and already
held keys are ignored, and saving/loading is guarded against concurrent actions.
Shortcuts are inactive in the notes field and other form inputs, during vertex
dragging/pointer editing, and while saving/loading. Modifier combinations are
ignored. Left/right navigation only changes the viewed case and flushes any draft.

U appends a `REOPEN` event for the most recently saved decision that is still
active (across reloads and roles), restores its contour/notes/checks as a pending
draft, and opens that case. Repeating U with separate presses moves to earlier
active decisions. The original decision, timestamp and history artifact remain
intact. A stale undo request is rejected if another decision was saved meanwhile.
For VAL/TEST, U records a fixed user-undo reason, hides predictions, preserves the
first blind-label hash, and retains model-exposed status. The separate reasoned
unlock action is also available.

Use **Prvi nedovršeni** to return to the first remaining pending case. With fast
review disabled, a VAL/TEST decision stays on the locked case to inspect metrics.

## Editing and persistence

- Click to add a vertex; polygons close automatically at three vertices.
- Drag vertices to move them. Shift-click near an edge to insert a vertex.
- Right-click a vertex, or select it and press Delete, to remove it.
- Z undoes a geometry edit. Separate buttons clear contours, create another
  contour, reset to the TRAIN candidate, and load a usable human reference.
- Mouse wheel zooms. Space-drag or the middle mouse button pans.
- Hide contour provides an unobstructed view of the original image.
- Geometry, notes, and confirmation changes autosave as drafts after a short
  idle period. Navigation flushes pending edits. Browser-local backup supports
  explicit recovery when a request fails or the page closes early.

The model panels show the **full saved binary masks**, including any additional
components and holes. The candidate editor starts with the largest external
contour produced by the frozen inference helper. These are distinct
representations; component, hole, and crop-edge flags are shown in TRAIN and
after VAL/TEST lock. Unchanged outlines that fail export geometry validation
can be recorded as accepted but are not training candidates.

## Blind VAL/TEST

Before locking, the server returns only the original/context images, neutral
identifiers, the user's annotation state, and genuine prior human-reference
information. Prediction masks are denied at the asset endpoint; prediction
polygons, confidence, model-derived historical warnings, and metrics are absent
from the case response.

The editor starts empty. A previous human reference may be loaded explicitly
only if its original coordinates map losslessly to the raw crop. Across the
34 prior references, 29 map losslessly and five extend outside the raw crop;
the latter cannot be imported. No clipping or scaling is applied.

Decision 2, 3, 4, or 5 saves and locks the human label. Only then are both model
masks and applicable positive-GT metrics revealed: IoU, Dice, pixel recall,
pixel precision, and predicted/GT area ratio. Metrics use the full saved masks.
Unlocking requires a recorded reason, hides predictions again, preserves the
first blind-label hash, and permanently marks subsequent revisions as
model-exposed. “Blind” here describes exposure within this tool.

## Stored records and restart

- `review.sqlite3`: event history with optimistic revision checks and triggers
  rejecting event UPDATE/DELETE.
- `annotation_revisions/<event_hash>.json`: exclusively created historical
  event artifacts, linked by a global hash chain.
- `logs/web_review.log`: server log.

On startup, the server verifies frozen inputs, prediction seals, the checkpoint
identity in configuration, and the existing review history.

```bash
python3 scripts/ruler_v7/web_controlled_ruler_review.py --port 8770
```

## Verification

```bash
python3 -m unittest discover -s scripts/ruler_v7 -p test_web_controlled_ruler_review.py -v
```

Ten isolated synthetic tests cover persisted drafts, optimistic conflicts,
append-only history and tamper detection, TRAIN eligibility, blind lock/unlock
and exposure provenance, geometry/reference restrictions, HTTP prediction
access, frozen-prediction tampering, explicit fast-review confirmations, and
append-only undo with conflict/exposure checks. JavaScript syntax and Python compilation
were checked. Headless Chrome checks exercised the real bundle using GET-only
requests and separately exercised editing, autosave, reload/resume, reference
confirmation, lock/reveal, and unlock on disposable synthetic data.

The pre-fast-review backup and proposed expanded-crop workflow are documented in
[`RULER_EXPANDED_CROP_MIGRATION_PLAN.md`](RULER_EXPANDED_CROP_MIGRATION_PLAN.md).
