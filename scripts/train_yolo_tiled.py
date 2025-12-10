#!/usr/bin/env python3
"""
Train YOLO on tiled dataset with multi-GPU support.

Usage:
    python scripts/train_yolo_tiled.py --device 0,1,2,3  # All 4 GPUs
    python scripts/train_yolo_tiled.py --device 0        # Single GPU
    python scripts/train_yolo_tiled.py --epochs 200      # Custom epochs
"""

import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch

def train_tiled_yolo(
    model_name='yolo11n.pt',
    data_yaml='data/yolo_tiled/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=-1,  # Auto batch size
    device='0,1,2,3',  # All 4 GPUs by default
    patience=30,
    project='runs/detect',
    name=None
):
    """Train YOLO on tiled dataset."""
    
    # Setup
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'train_tiled_{imgsz}_{timestamp}'
    
    # Check GPU availability
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"\nGPU Info:")
        print(f"  Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print(f"  Using devices: {device}")
    else:
        print("WARNING: No GPU available, training on CPU")
        device = 'cpu'
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'patience': patience,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'project': project,
        'name': name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': False,  # Faster training
        'single_cls': True,  # Single class (collembola)
        'rect': False,  # Don't use rectangular training (causes issues with DDP)
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for last 10 epochs
        'amp': True,  # Automatic Mixed Precision for faster training
        'fraction': 1.0,  # Use 100% of dataset
        'profile': False,
        'freeze': None,
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate fraction
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Class loss weight (lower for single class)
        'dfl': 1.5,  # DFL loss weight
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,  # HSV hue augmentation
        'hsv_s': 0.7,    # HSV saturation augmentation
        'hsv_v': 0.4,    # HSV value augmentation
        'degrees': 0.0,  # Rotation augmentation (degrees)
        'translate': 0.1,  # Translation augmentation (fraction)
        'scale': 0.5,    # Scale augmentation
        'shear': 0.0,    # Shear augmentation (degrees)
        'perspective': 0.0,  # Perspective augmentation
        'flipud': 0.0,   # Vertical flip probability
        'fliplr': 0.5,   # Horizontal flip probability
        'mosaic': 1.0,   # Mosaic augmentation probability
        'mixup': 0.0,    # Mixup augmentation probability
        'copy_paste': 0.0,  # Copy-paste augmentation probability
    }
    
    print(f"\nTraining configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch} (auto)")
    print(f"  Device: {device}")
    print(f"  Patience: {patience}")
    print(f"  Output: {project}/{name}")
    
    # Train
    print("\nStarting training...")
    results = model.train(**train_args)
    
    # Validation
    print("\nRunning validation...")
    metrics = model.val()
    
    # Print results
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Results saved to: {project}/{name}")
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"\nValidation Metrics:")
    print(f"  Precision: {metrics.box.p[0]:.3f}")
    print(f"  Recall:    {metrics.box.r[0]:.3f}")
    print(f"  mAP@0.5:   {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    print("="*70)
    
    return model, results, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO on tiled collembola dataset')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        help='Model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)')
    parser.add_argument('--data', type=str, default='data/yolo_tiled/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size for training')
    parser.add_argument('--batch', type=int, default=-1,
                        help='Batch size (-1 for auto)')
    parser.add_argument('--device', type=str, default='0,1,2,3',
                        help='GPU devices to use (e.g., "0" or "0,1,2,3")')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    train_tiled_yolo(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name
    )
