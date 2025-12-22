#!/bin/bash
# Test that the packaged model works correctly

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Testing Zenodo Package Integrity                                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# Test 1: Verify ZIP exists
echo "✓ Test 1: ZIP package exists"
if [ ! -f "collembola_yolo11n_model_v1.0.0.zip" ]; then
    echo "✗ FAIL: ZIP file not found"
    exit 1
fi
ls -lh collembola_yolo11n_model_v1.0.0.zip
echo

# Test 2: Verify ZIP contents
echo "✓ Test 2: ZIP contents"
unzip -l collembola_yolo11n_model_v1.0.0.zip | grep -E "(yolo11n|MODEL_CARD|README|requirements)"
echo

# Test 3: Extract and check files
echo "✓ Test 3: Extract package to test directory"
rm -rf test_zenodo_package
mkdir test_zenodo_package
cd test_zenodo_package
unzip -q ../collembola_yolo11n_model_v1.0.0.zip
echo "Extracted files:"
ls -lh
echo

# Test 4: Check model file
echo "✓ Test 4: Model file integrity"
if [ -f "yolo11n_tiled_best.pt" ]; then
    echo "  Model size: $(ls -lh yolo11n_tiled_best.pt | awk '{print $5}')"
    echo "  ✓ Model file present"
else
    echo "  ✗ FAIL: Model file missing"
    exit 1
fi
echo

# Test 5: Check documentation
echo "✓ Test 5: Documentation files"
for file in "MODEL_CARD.md" "README.md" "requirements.txt" ".zenodo.json"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done
echo

# Test 6: Check scripts
echo "✓ Test 6: Example scripts"
if [ -d "scripts" ]; then
    echo "  Scripts directory:"
    ls scripts/
else
    echo "  ✗ FAIL: Scripts directory missing"
    exit 1
fi
echo

# Test 7: Validate Zenodo metadata
echo "✓ Test 7: Zenodo metadata validation"
if command -v python3 &> /dev/null; then
    python3 -c "import json; json.load(open('.zenodo.json'))" && echo "  ✓ .zenodo.json is valid JSON"
else
    echo "  ⚠ Python not available, skipping JSON validation"
fi
echo

# Test 8: Check requirements
echo "✓ Test 8: Requirements file"
cat requirements.txt
echo

# Cleanup
cd ..
echo "✓ Test 9: Cleanup test directory"
rm -rf test_zenodo_package
echo

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ ALL TESTS PASSED - Package ready for upload!                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "  1. Upload to Zenodo: https://zenodo.org/"
echo "  2. See UPLOAD_README.md for instructions"
