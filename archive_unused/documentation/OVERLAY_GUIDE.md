# 🎨 Overlay Generation Guide

## Problem: No Overlays Created

Your script ran successfully and created measurements, but **no overlay was generated** because the `--save-overlay` flag was not specified in your command.

## Solution: Add the Overlay Flag

### Option 1: Use the Pre-configured Example Script ✅ RECOMMENDED
```bash
./run_example.sh
```
This automatically includes all output flags:
- `--save-overlay out/overlay.png`
- `--save-masks-dir out/masks`
- `--output out/measurements.csv`
- `--json out/summary.json`

### Option 2: Run Quick Test
```bash
./test_with_overlay.sh
```
Generates overlay in `out/test/overlay.png`

### Option 3: Manual Command with Overlay
```bash
python sam_templates.py "data/slike/K1_Fe2O3001 (1).jpg" \
    --template-dir data/organism_templates \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --sam-model-type vit_b \
    --output out/measurements.csv \
    --save-overlay out/overlay.png \      # ← ADD THIS LINE
    --save-masks-dir out/masks \          # ← AND THIS LINE
    --auto-download \
    --allow-large-image
```

## Output Files Explained

| Flag | Output | Description |
|------|--------|-------------|
| `--output` | CSV file | Per-specimen measurements (always recommended) |
| `--json` | JSON file | Complete summary with metadata |
| `--save-overlay` | PNG image | **Visualization with colored masks** ← YOU NEED THIS! |
| `--save-masks-dir` | Directory | Individual binary mask PNGs |

## What the Overlay Looks Like

The overlay is a visualization showing:
- Original microscope image as background
- Each detected collembola highlighted with a **random color**
- Easy visual verification of detection quality

Example overlay features:
- 🔴 Red specimens
- 🟢 Green specimens
- 🔵 Blue specimens
- 🟡 Yellow specimens
- etc. (colors are randomized)

## Your Current Run

Based on `out/measurements.csv`, you detected **7 collembolas** but:
- ❌ No overlay saved (missing `--save-overlay` flag)
- ❌ No masks saved (missing `--save-masks-dir` flag)
- ✅ Measurements CSV created

## Next Steps

1. **Run with overlay enabled:**
   ```bash
   ./run_example.sh
   ```

2. **View the overlay:**
   ```bash
   # Linux with GNOME
   eog out/overlay.png
   
   # Linux with generic viewer
   xdg-open out/overlay.png
   
   # Or use any image viewer
   ```

3. **Expected overlay location:**
   ```
   out/
   ├── overlay.png          ← YOUR VISUALIZATION HERE
   ├── measurements.csv
   ├── summary.json
   └── masks/
       ├── template_mask_1.png
       ├── template_mask_2.png
       └── ...
   ```

## Troubleshooting

### "No overlay after running"
**Check:** Did you include `--save-overlay out/overlay.png` in the command?

### "Overlay is black/empty"
**Possible causes:**
- No collembolas detected (too strict thresholds)
- Lower `--ncc-threshold` (e.g., from 0.65 to 0.5)
- Increase `--scale-factors` to `0.75,1.0,1.25`

### "Out of memory during overlay creation"
**Solutions:**
- Use smaller `--downscale-max-side` (e.g., 1024 instead of 2048)
- Overlay creation is fast, so this is unlikely

## Code Reference

Overlay generation code in `sam_templates.py`:
```python
# Line 722-727
if args.save_overlay and masks:
    print(f"\n🎨 Creating overlay visualization...")
    args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay = build_overlay(image, masks)
    io.imsave(str(args.save_overlay), overlay)
    print(f"✓ Overlay saved: {args.save_overlay}")
```

The overlay will **only** be created if:
1. You specify `--save-overlay <path>`
2. At least one collembola is detected (`masks` is not empty)

---

**TL;DR:** Run `./run_example.sh` to get your overlay! 🎨
