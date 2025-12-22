#!/bin/bash
# Package model and documentation for Zenodo upload

set -e

echo "Creating Zenodo upload package..."

# Create upload directory
UPLOAD_DIR="zenodo_upload"
rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR/scripts"

# Copy model
echo "Copying model file..."
cp models/yolo11n_tiled_best.pt "$UPLOAD_DIR/"

# Copy documentation
echo "Copying documentation..."
cp MODEL_CARD.md "$UPLOAD_DIR/"
cp README.md "$UPLOAD_DIR/"
cp PERFORMANCE.md "$UPLOAD_DIR/"
cp QUICKSTART.md "$UPLOAD_DIR/"
cp .zenodo.json "$UPLOAD_DIR/"
cp LICENSE "$UPLOAD_DIR/" 2>/dev/null || echo "LICENSE file not found, skipping..."

# Copy example scripts
echo "Copying example scripts..."
cp scripts/infer_tiled.py "$UPLOAD_DIR/scripts/"
cp scripts/measure_organisms_fast.py "$UPLOAD_DIR/scripts/"
cp scripts/process_plate_batch.py "$UPLOAD_DIR/scripts/" 2>/dev/null || echo "process_plate_batch.py not found"

# Create requirements.txt for easy setup
echo "Creating requirements.txt..."
cat > "$UPLOAD_DIR/requirements.txt" << EOF
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.23.0
opencv-python>=4.7.0
pandas>=1.5.0
tqdm>=4.64.0
EOF

# Create simple inference example
echo "Creating example script..."
cat > "$UPLOAD_DIR/example_inference.py" << 'EOF'
"""
Simple example: Run YOLO11n collembola detection on an image.

Usage:
    python example_inference.py --image path/to/image.jpg
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Run collembola detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--save', default='output', help='Output directory')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = YOLO('yolo11n_tiled_best.pt')
    
    # Run detection
    print(f"Running detection on {args.image}...")
    results = model(args.image, conf=args.conf, save=True, project=args.save)
    
    # Print results
    detections = results[0].boxes
    print(f"Found {len(detections)} organisms")
    print(f"Results saved to {args.save}")

if __name__ == '__main__':
    main()
EOF

# Create archive
echo "Creating ZIP archive..."
cd "$UPLOAD_DIR"
zip -r ../collembola_yolo11n_model_v1.0.0.zip .
cd ..

# Print summary
echo ""
echo "✓ Package created successfully!"
echo ""
echo "Files included:"
echo "  - yolo11n_tiled_best.pt (5.4 MB)"
echo "  - MODEL_CARD.md"
echo "  - README.md"
echo "  - PERFORMANCE.md"
echo "  - QUICKSTART.md"
echo "  - .zenodo.json"
echo "  - requirements.txt"
echo "  - example_inference.py"
echo "  - scripts/infer_tiled.py"
echo "  - scripts/measure_organisms_fast.py"
echo ""
echo "Archive: collembola_yolo11n_model_v1.0.0.zip"
ls -lh collembola_yolo11n_model_v1.0.0.zip
echo ""
echo "Next steps:"
echo "  1. Upload to Zenodo: https://zenodo.org/"
echo "  2. See UPLOAD_INSTRUCTIONS.md for details"
