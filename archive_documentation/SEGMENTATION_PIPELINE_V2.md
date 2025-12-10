# New Collembola Segmentation Pipeline V2

## Overview

**Goal**: Detect and segment collembolas in any plate image from `data/slike/` without manual annotation.

**Input**: Raw plate image (10K×10K pixels)  
**Output**: 
- Individual organism masks (precise pixel-level segmentation)
- Bounding boxes
- Morphological measurements (length, width, area, volume)
- CSV export

---

## Pipeline Architecture

### Stage 1: Region Proposal (GPU-Accelerated)
**Purpose**: Find candidate regions that might contain organisms

**Options:**

#### Option A: SAM AutoMask (Current approach)
- ✅ **Pros**: Already implemented, GPU-accelerated, finds everything
- ❌ **Cons**: Slow (~5-10 min per plate), generates 1200-1400 proposals (many junk)
- **Keep if**: We want universal segmentation without tuning

#### Option B: CV Preprocessing + Blob Detection (Faster alternative)
- ✅ **Pros**: Much faster (<1 min), GPU-accelerated with cupy/opencv-cuda
- ✅ **Pros**: Fewer proposals (200-400 candidates vs 1200+)
- ❌ **Cons**: Needs parameter tuning per plate type
- **Techniques**:
  1. **Background subtraction**: Morphological opening to remove background
  2. **Contrast enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. **Thresholding**: Otsu or adaptive threshold
  4. **Blob detection**: SimpleBlobDetector or contour detection
  5. **Size filtering**: Remove too-small (<40px) and too-large (>600px) blobs

**Recommendation**: Start with **Option B** (CV preprocessing), fall back to SAM if recall is poor.

---

### Stage 2: CNN Classification
**Purpose**: Filter out junk, keep only organism regions

**Model**: ResNet18 (already trained! 98.6% F1 score)
- Input: 224×224 RGB crop from proposal bbox
- Output: p(organism) confidence score
- Threshold: 0.97 (configurable)

**Fast inference**: Batch process all crops at once (64 crops/batch on GPU)

---

### Stage 3: Precise Segmentation (Watershed)
**Purpose**: Get exact pixel-level organism boundary from accepted regions

**For each accepted proposal:**

1. **Extract ROI** from full image using bbox
2. **Preprocessing**:
   - Convert to grayscale
   - Gaussian blur (σ=1) to reduce noise
   - CLAHE contrast enhancement
3. **Marker-based Watershed**:
   - **Foreground markers**: Distance transform + local maxima
   - **Background markers**: Dilated border
   - Run watershed to separate touching organisms
4. **Mask refinement**:
   - Fill holes (morphological closing)
   - Remove small artifacts (area < 200px)
   - Smooth boundaries (morphological opening)
5. **Extract measurements**:
   - Fit ellipse → major/minor axes (length/width)
   - Compute area, perimeter, eccentricity, solidity
   - Calculate volume (prolate spheroid)

---

## Implementation Plan

### Phase 1: CV-Based Region Proposal (New)
**File**: `collembola_pipeline/proposal_cv.py`

```python
def propose_regions_cv(
    image: np.ndarray,
    min_area: int = 200,
    max_area: int = 20000,
    min_eccentricity: float = 0.7
) -> List[Dict]:
    """
    Fast region proposal using classical CV.
    
    Returns list of proposals: [{'bbox': (x,y,w,h), 'mask': binary_array}, ...]
    """
```

**Steps**:
1. Convert to grayscale
2. Background subtraction (morphological opening, large kernel)
3. CLAHE contrast enhancement
4. Adaptive threshold or Otsu
5. Find contours
6. Filter by size + eccentricity
7. Return bboxes + masks

### Phase 2: Batch CNN Classification (Optimize existing)
**File**: `collembola_pipeline/classify_batch.py`

```python
def classify_batch(
    image: np.ndarray,
    proposals: List[Dict],
    model,
    device: str = 'cuda',
    batch_size: int = 64,
    threshold: float = 0.97
) -> List[int]:
    """
    Classify all proposals in batches.
    
    Returns indices of accepted proposals.
    """
```

**Optimization**: Process all crops in parallel batches instead of one-by-one.

### Phase 3: Watershed Segmentation (New)
**File**: `collembola_pipeline/segment_watershed.py`

```python
def refine_mask_watershed(
    image: np.ndarray,
    bbox: Tuple[int,int,int,int],
    rough_mask: np.ndarray = None
) -> np.ndarray:
    """
    Apply watershed to get precise organism boundary.
    
    Args:
        image: Full plate image
        bbox: Region of interest (x, y, w, h)
        rough_mask: Optional initial mask from region proposal
    
    Returns:
        Refined binary mask (full image coordinates)
    """
```

### Phase 4: Orchestration
**File**: `collembola_pipeline/detect_organisms.py`

```python
def detect_organisms(
    plate_path: Path,
    device: str = 'cuda',
    use_sam: bool = False,  # False = CV proposal, True = SAM
    verbose: bool = False
) -> List[Dict]:
    """
    Complete detection pipeline.
    
    Returns list of organism detections with masks and measurements.
    """
    # 1. Load image
    # 2. Propose regions (CV or SAM)
    # 3. Batch classify
    # 4. Refine with watershed
    # 5. Extract measurements
    # 6. Return results
```

---

## Expected Performance

### Speed (on 10K×10K plate):
- **CV Proposal**: ~30 seconds (vs SAM: 5-10 min)
- **CNN Classification**: ~2 seconds for 400 proposals in batches
- **Watershed refinement**: ~10 seconds for 700 accepted regions
- **Total**: **~45 seconds per plate** (10x faster than current SAM-based)

### Accuracy:
- **Recall**: 80-90% (CV proposal might miss some)
- **Precision**: 75-85% (CNN filters most junk)
- **Segmentation quality**: Better than bounding boxes (pixel-accurate boundaries)

---

## Fallback Strategy

If CV proposal has poor recall:
1. Keep SAM proposal option available (`use_sam=True`)
2. Hybrid approach: CV for "easy" organisms, SAM for missed regions
3. Ensemble: Merge CV + SAM proposals, deduplicate with NMS

---

## File Structure

```
collembola_pipeline/
├── proposal_cv.py          # NEW: Classical CV region proposal
├── proposal_sam.py         # Refactor from segment.py
├── classify_batch.py       # NEW: Batch CNN classification
├── segment_watershed.py    # NEW: Watershed refinement
├── detect_organisms.py     # NEW: Main orchestration
├── morphology.py           # Existing: Measurement extraction
├── visualize.py            # Existing: Overlay generation
└── config.py               # Update with new params
```

---

## Next Steps

1. Implement `proposal_cv.py` with blob detection
2. Test on one plate, compare proposal count vs SAM
3. Implement `classify_batch.py` for fast inference
4. Implement `segment_watershed.py` for refinement
5. Integrate into `detect_organisms.py`
6. Benchmark speed and accuracy vs current pipeline
7. Tune parameters if needed

---

## Open Questions

1. **Which CV preprocessing gives best proposals?**
   - Need to test: Otsu vs adaptive threshold, different kernel sizes
   
2. **Is watershed necessary or is bbox+ellipse-fit enough?**
   - Depends on downstream analysis needs
   
3. **Should we use the trained CNN model (with contaminated hard negatives)?**
   - Yes - test it first, retrain only if poor performance
