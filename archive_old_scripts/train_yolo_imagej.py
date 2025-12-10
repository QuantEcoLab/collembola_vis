#!/usr/bin/env python3
"""
Train YOLO model on ImageJ ROI dataset.

This script trains a YOLOv11 (or YOLOv8) model on the collembola dataset
extracted from ImageJ ROI annotations.

The images are ~10K x 10K pixels, so we use appropriate image size settings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import torch
import argparse
from datetime import datetime


def train_yolo(
    data_yaml: Path,
    model_name: str = 'yolo11n.pt',
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = -1,
    device: str = '0',
    project: str = 'runs/detect',
    name: str = None,
    resume: bool = False,
    patience: int = 50,
    save_period: int = 10,
):
    """
    Train YOLO model on collembola dataset.
    
    Args:
        data_yaml: Path to data.yaml config
        model_name: Model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size (images will be resized to imgsz x imgsz)
        batch: Batch size (-1 for auto)
        device: GPU device(s) to use (e.g., '0' or '0,1,2,3')
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    print("="*70)
    print("YOLO Training for Collembola Detection")
    print("="*70)
    print()
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"ERROR: Data config not found: {data_yaml}")
        print(f"Please run: python scripts/imagej_rois_to_yolo.py first")
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Information:")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
    else:
        print("WARNING: No GPU detected. Training will be slow on CPU.")
        print()
    
    # Auto-generate name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"train_{model_name.replace('.pt', '')}_{imgsz}_{timestamp}"
    
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch: {batch} {'(auto)' if batch == -1 else ''}")
    print(f"  Device: {device}")
    print(f"  Project: {project}")
    print(f"  Name: {name}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Save period: {save_period}")
    print()
    
    # Load model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    print()
    
    # Train
    print("Starting training...")
    print("="*70)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        patience=patience,
        save_period=save_period,
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation (careful with large images)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation (organisms are directional)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Training settings
        cos_lr=True,
        close_mosaic=10,
        # Validation
        val=True,
        plots=True,
        save=True,
        save_json=False,
        cache=False,  # Don't cache (images are huge)
        rect=False,
        # Advanced
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        verbose=True,
    )
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print()
    
    # Print results location
    save_dir = Path(project) / name
    print(f"Results saved to: {save_dir}")
    print(f"  Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"  Last weights: {save_dir / 'weights' / 'last.pt'}")
    print(f"  Metrics: {save_dir / 'results.csv'}")
    print(f"  Plots: {save_dir / 'results.png'}")
    print()
    
    # Show best metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("Best Validation Metrics:")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        if 'metrics/precision(B)' in metrics:
            print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        print()
    
    print("Next steps:")
    print(f"  1. Validate: yolo detect val model={save_dir}/weights/best.pt data={data_yaml}")
    print(f"  2. Predict: yolo detect predict model={save_dir}/weights/best.pt source=data/slike/")
    print()
    
    return results, save_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model on ImageJ ROI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with default settings (YOLO11n, 1280px, 100 epochs)
  python scripts/train_yolo_imagej.py
  
  # Larger model with higher resolution
  python scripts/train_yolo_imagej.py --model yolo11m.pt --imgsz 1920 --epochs 150
  
  # Multi-GPU training
  python scripts/train_yolo_imagej.py --device 0,1,2,3
  
  # Resume training from checkpoint
  python scripts/train_yolo_imagej.py --resume
  
  # Custom batch size
  python scripts/train_yolo_imagej.py --batch 8
        """
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/yolo_imagej/data.yaml'),
        help='Path to data.yaml config (default: data/yolo_imagej/data.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='Model name (default: yolo11n.pt). Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1280,
        help='Image size for training (default: 1280). Options: 640, 1280, 1920, 2560'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=-1,
        help='Batch size (default: -1 for auto)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device(s) to use (default: 0). Use 0,1,2,3 for multi-GPU'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Train
    results, save_dir = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        save_period=args.save_period,
    )


if __name__ == '__main__':
    main()
