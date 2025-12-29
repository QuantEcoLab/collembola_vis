# Performance Benchmarks

## Detection Performance

### YOLO Tiled Inference (K1 plate - 10408×10338 pixels)

| Confidence Threshold | Detections Before NMS | Detections After NMS | Time |
|---------------------|----------------------|---------------------|------|
| 0.25 (default) | 1,214 | 800 | ~2-3 min |
| **0.6 (recommended)** | **1,117** | **746** | **~2-3 min** |

**Recommendation**: Use `--conf 0.6` to filter out low-confidence false positives while maintaining high recall.

## Measurement Performance

### K1 Plate Test (746 organisms with conf ≥ 0.6)

| Method | Time | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| **Fast Ellipse** ⚡ | **4.2 sec** | **178 org/sec** | Good (ellipse approximation) | **Production, batch processing** |
| SAM Segmentation | ~13 min | 1 org/sec | Best (precise masks) | Research, validation |

**Speed Improvement**: Fast ellipse method is **186× faster** than SAM!

### Measurement Results Comparison

**Fast Ellipse Method (K1 plate, 746 organisms):**
```
Length (µm):  Mean: 878.9,  Median: 685.9,  Range: 100.3-3504.1
Width (µm):   Mean: 232.4,  Median: 170.3
Volume (µm³): Mean: 124M,   Median: 16.2M,  Total: 92.5B
```

**Method Details:**
- Adaptive thresholding on cropped bbox
- Fit ellipse to largest connected component
- Extract major/minor axes for length/width
- Cylinder volume: V = π × r² × h

## Complete Pipeline Timing

### Single Plate (10K×10K, ~750 organisms)

| Step | Time | Method |
|------|------|--------|
| 1. YOLO Detection | ~2-3 min | Tiled inference (100 tiles) |
| 2. Fast Measurement | ~4 sec | Ellipse fitting |
| **Total** | **~3 min** | **End-to-end** |

### Batch Processing (20 plates)

| Component | Time | Throughput |
|-----------|------|------------|
| Detection (20 plates) | ~50 min | 2.5 min/plate |
| Measurement (15,000 organisms) | ~80 sec | 188 org/sec |
| **Total** | **~52 min** | **2.6 min/plate** |

## Hardware Requirements

**Tested Configuration:**
- GPU: NVIDIA Quadro RTX 8000 (48GB VRAM)
- RAM: 64GB
- Storage: SSD recommended for large images

**Minimum Requirements:**
- GPU: Any CUDA-capable GPU with 8GB+ VRAM (for detection)
- RAM: 16GB+ (for loading 10K×10K images)
- Storage: ~50MB per plate image

## Scalability

**Processing 1,000 plates:**
- Detection: ~50 hours (2-3 min/plate)
- Measurement: ~2.2 hours (80 sec/20 plates)
- **Total**: ~52 hours (~3 min/plate average)

**With 4× GPUs (parallel batches):**
- Can reduce to ~13 hours total

## Optimization Tips

1. **Use conf=0.6** to reduce false positives
2. **Use fast measurement** for production (186× faster)
3. **Batch process** multiple plates to amortize overhead
4. **Use SSD storage** for faster image loading
5. **Pre-calibrate** ruler once per microscope setup

## Memory Usage

| Component | RAM | VRAM |
|-----------|-----|------|
| YOLO Detection | ~2GB | ~4GB |
| Fast Measurement | ~2GB | 0GB (CPU) |
| SAM Measurement | ~3GB | ~6GB |
| Full Image Load | ~1GB | - |

**Note**: Processing 10K×10K images requires ~1GB RAM per image loaded.
