# Archive Notes - Template Matching Approach (Nov 29, 2025)

## What We Tried

### Approach: Template-Guided SAM Segmentation
- Used 214 organism crops as templates
- Normalized Cross-Correlation (NCC) matching across multiple scales
- SAM segmentation using template matches as prompts

### Performance Issues
1. **Speed**: ~10 minutes per plate (214 templates × 5 scales = 1070 iterations)
2. **Recall**: Only 7/692 organisms detected (1% recall) - UNACCEPTABLE
3. **Scalability**: Doesn't generalize to new organism variations

### What We Built
- `sam_templates.py` - Main detection script with template matching
- `detect_auto.sh` - Shell script with optimized parameters
- `detect_all.sh` - Fast detection with 50 templates
- `run_example.sh` - Example usage
- `test_with_overlay.sh` - Quick overlay test
- Documentation: `CHANGELOG.md`, `DETECTION_METHODS.md`, `OVERLAY_GUIDE.md`, `README.md`

## Why It Failed

### Fundamental Issues
1. **Wrong Assumption**: Template matching assumes organisms look nearly identical
2. **Reality**: High biological variation in:
   - Size (50-200 pixels)
   - Orientation (0-360°)
   - Pose (curled, stretched, twisted)
   - Appearance (lighting, focus, occlusion)

3. **Computational Cost**: O(N_templates × N_scales × Image_size)
   - Even with optimizations, doesn't scale to 214 templates

4. **Brittleness**: Small changes in NCC threshold drastically affect results
   - threshold=0.65 → 7 detections
   - threshold=0.50 → unknown (script too slow to test)

## Lessons Learned

### What Worked
- ✅ Progress bars and status messages (good UX)
- ✅ Auto-downscaling for large images (2048px max)
- ✅ Overlay visualization for debugging
- ✅ CSV export format matches downstream needs
- ✅ `volumen.py` integration for volume calculation

### What Didn't Work
- ❌ Template matching for high-variation objects
- ❌ Treating templates as "prototypes" (they're just training data)
- ❌ Trying to optimize a fundamentally wrong approach

## Files to Archive

### Scripts
- `sam_templates.py` (modified version with --max-templates)
- `detect_auto.sh`
- `detect_all.sh`
- `run_example.sh`
- `test_with_overlay.sh`

### Documentation
- `CHANGELOG.md` (template approach history)
- `DETECTION_METHODS.md` (comparison of methods)
- `OVERLAY_GUIDE.md` (how to create visualizations)
- Current `README.md` (will be replaced)

### Keep
- `volumen.py` (volume calculation - still useful)
- `sam_detect.py` (already archived - annotation-guided)
- `data/` (all data assets)
- `archive_old_scripts/` (previous archives)

## Why the New Approach Will Work

### SAM AutoMask + Classifier
1. **Separation of Concerns**:
   - SAM: Find ALL objects (high recall)
   - Classifier: Filter to organisms (high precision)

2. **Learned Features**:
   - ResNet learns what makes an organism an organism
   - Handles variation naturally through training data
   - Can improve by adding more examples

3. **Proven Architecture**:
   - Similar to R-CNN, Faster R-CNN, Mask R-CNN
   - Used in production systems worldwide
   - Well-understood failure modes and solutions

4. **Fast**:
   - SAM AutoMask: ~1-2 min per plate
   - ResNet inference: ~1ms per crop × 1000 crops = 1 sec
   - Total: ~2-3 min per plate (3-5x faster than template matching)

## Archive Location

All files moved to: `archive_template_approach/`
Created: 2025-11-29
Reason: Fundamental approach failure - switching to SAM AutoMask + Classifier

---

**Status**: Template matching archived, ready for new implementation
