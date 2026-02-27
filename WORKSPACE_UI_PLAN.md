# Workspace UI Ergonomics — Dev Plan

## Scope
Full ergonomics pass on WorkspacePage. All features tracked here.

---

## Features

| # | Feature | File(s) | Status |
|---|---------|---------|--------|
| 1 | Overlay toggle (Raw / Detection / SAM) | WorkspacePage | DONE |
| 2 | Overlay label badge in viewer | WorkspacePage | DONE |
| 3 | Resizable viewer/table split (drag handle) | WorkspacePage | DONE |
| 4 | Table row → box selection (bidirectional) | WorkspacePage, MeasurementTable | DONE |
| 5 | Column visibility toggle in table | MeasurementTable | DONE |
| 6 | Measurement summary bar above table | WorkspacePage | DONE |
| 7 | Confidence range slider | WorkspacePage | DONE |
| 8 | Keyboard shortcuts: S/D/H/Esc + hints in toolbar | WorkspacePage | DONE |
| 9 | Sidebar collapse button | WorkspacePage | DONE |
| 10 | Workflow stepper at top of sidebar | WorkspacePage | DONE |
| 11 | Persist umPerPixel on manual type | WorkspacePage, calibrationStore | DONE |
| 12 | Persist rulerMm across sessions | calibrationStore, WorkspacePage | DONE |
| 13 | Add persist middleware to calibrationStore | calibrationStore | DONE |

---

## Implementation Order

### Pass 1 — calibrationStore.ts
- Add `persist` middleware (currently not persisted at all!)
- Add `rulerMm: number` field (default 10) + `setRulerMm`
- Add `setUmManual(um)` for manual input persistence

### Pass 2 — MeasurementTable.tsx
- Add `onRowClick?: (originalIndex: number) => void` prop
- Add TanStack column visibility state, initialized from DEFAULT_HIDDEN set
- Add "Columns ▾" button + popover checklist
- Make rows clickable with cursor-pointer when onRowClick provided

DEFAULT_HIDDEN columns:
  bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_width_px, bbox_height_px,
  centroid_x_px, centroid_y_px, area_px, perimeter_px,
  major_axis_px, minor_axis_px, eccentricity, solidity, confidence, method

### Pass 3 — WorkspacePage.tsx (large rewrite)

**New state:**
- `viewMode: 'raw' | 'detection' | 'sam'` — drives viewerSrc outside refineMode
- `splitPercent: number` (default 58) — viewer/table vertical split
- `sidebarCollapsed: boolean` — sidebar collapse

**Removed state:**
- `rulerMm` local state → replaced by calibrationStore.rulerMm

**Modified logic:**
- `viewerSrc`: uses viewMode instead of auto-detecting
- `umPerPixel` onChange: also calls calibrationStore.setUmManual(value)
- Merge Delete/Backspace + new S/D/H/Esc into single keydown handler
- Auto-advance viewMode: detection done → 'detection'; SAM done → 'sam'
- `handleReset`: reset viewMode to 'raw', splitPercent to 58

**New handlers:**
- `onSplitPointerDown`: document-level pointermove/pointerup for drag
- `handleRowClick(originalIndex)`: enter refineMode + selectBox

**New JSX sections:**
- Overlay toggle buttons (top-left of viewer): Raw / Detection / SAM
- Overlay label badge (bottom-left of viewer): what's currently shown
- Drag handle between viewer and table (h-1.5, cursor-row-resize)
- Measurement summary bar (length_mm, width_mm, area_mm2, volume_mm3 mean+range)
- Confidence replaced with <input type="range"> + live numeric display
- Sidebar collapse button (ChevronLeft/Right on sidebar edge)
- Workflow stepper (4 steps: Scale, Detect, Annotate, Measure) at sidebar top
- Keyboard hints in floating annotation toolbar: [S] [D] [H] [Del] [Esc]

---

## Key Column Names (from measure_organisms_fast.py)
detection_id | bbox_x1..y2 | bbox_width_px | bbox_height_px |
centroid_x_px | centroid_y_px | length_mm | width_mm | area_mm2 | volume_mm3 |
area_px | perimeter_px | major_axis_px | minor_axis_px | eccentricity | solidity |
confidence | method

---

## Done
(items moved here as completed)
