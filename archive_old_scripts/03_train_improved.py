"""
Improved ResNet18 classifier training with comprehensive metrics and logging.

Improvements over original 03_train.py:
- Larger input size (224x224 instead of 128x128)
- Class balancing via WeightedRandomSampler  
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on F1 score
- Comprehensive metrics (precision, recall, F1, confusion matrix)
- Training logs saved to file
- Best model selection based on F1, not just accuracy
- Optional focal loss for hard example mining
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime

from collembola_pipeline.config import MODELS_DIR, CLASSIFIER_PATH


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights [class0_weight, class1_weight]
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CropDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool = False, size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.size = size
        if train:
            self.t = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)  # Must be after ToTensor
            ])
        else:
            self.t = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row['img_path'])
        label = int(row['label'])
        img = Image.open(path).convert('RGB')
        return self.t(img), label


def get_class_weights(dataset: Dataset) -> torch.Tensor:
    """Compute inverse frequency class weights"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def get_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Create weighted sampler for class balancing"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1, and confusion matrix"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def train(
    train_csv: Path,
    val_csv: Path,
    device='cuda',
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-5,
    input_size: int = 224,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_balancing: bool = True,
    early_stop_patience: int = 7,
    log_dir: Path = None
):
    # Setup logging
    if log_dir is None:
        log_dir = MODELS_DIR / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log(f"{'='*80}")
    log(f"Training Started: {timestamp}")
    log(f"{'='*80}")
    log(f"Config:")
    log(f"  Device: {device}")
    log(f"  Epochs: {epochs}")
    log(f"  Batch size: {batch_size}")
    log(f"  Learning rate: {lr}")
    log(f"  Input size: {input_size}x{input_size}")
    log(f"  Use focal loss: {use_focal_loss}")
    log(f"  Use class balancing: {use_class_balancing}")
    log(f"  Early stop patience: {early_stop_patience}")
    log(f"")
    
    # Load datasets
    ds_tr = CropDataset(train_csv, train=True, size=input_size)
    ds_va = CropDataset(val_csv, train=False, size=input_size)
    
    log(f"Dataset sizes:")
    log(f"  Training: {len(ds_tr)} samples")
    log(f"  Validation: {len(ds_va)} samples")
    
    # Compute class distribution
    train_labels = [ds_tr[i][1] for i in range(len(ds_tr))]
    val_labels = [ds_va[i][1] for i in range(len(ds_va))]
    train_pos = sum(train_labels)
    val_pos = sum(val_labels)
    
    log(f"  Training - Positives: {train_pos} ({train_pos/len(ds_tr)*100:.1f}%), Negatives: {len(ds_tr)-train_pos}")
    log(f"  Validation - Positives: {val_pos} ({val_pos/len(ds_va)*100:.1f}%), Negatives: {len(ds_va)-val_pos}")
    log(f"")
    
    # Setup data loaders
    if use_class_balancing:
        sampler = get_weighted_sampler(ds_tr)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        log("Using WeightedRandomSampler for class balancing")
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.to(device)
    
    log(f"Model: ResNet18 with {sum(p.numel() for p in model.parameters())} parameters")
    log(f"")
    
    # Setup loss
    if use_focal_loss:
        class_weights = get_class_weights(ds_tr).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        log(f"Using Focal Loss (gamma={focal_gamma}, alpha={class_weights.cpu().numpy()})")
    else:
        class_weights = get_class_weights(ds_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        log(f"Using Cross Entropy Loss (weights={class_weights.cpu().numpy()})")
    log(f"")
    
    # Setup optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    log(f"{'='*80}")
    log(f"Starting training...")
    log(f"{'='*80}")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_loss /= len(val_targets)
        val_metrics = compute_metrics(val_targets, val_preds)
        val_acc = sum(np.array(val_preds) == np.array(val_targets)) / len(val_targets)
        
        # Log epoch results
        log(f"Epoch {epoch:3d}/{epochs} | "
            f"Loss: train={train_loss:.4f} val={val_loss:.4f} | "
            f"Acc: train={train_acc:.3f} val={val_acc:.3f} | "
            f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'confusion_matrix': val_metrics['confusion_matrix'],
            'lr': opt.param_groups[0]['lr']
        })
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), CLASSIFIER_PATH)
            log(f"  *** New best F1: {best_f1:.3f} - Model saved to {CLASSIFIER_PATH}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            log(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
            log(f"Best F1: {best_f1:.3f} at epoch {best_epoch}")
            break
    
    # Save training history
    history_file = log_dir / f"history_{timestamp}.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    log(f"\n{'='*80}")
    log(f"Training Complete!")
    log(f"{'='*80}")
    log(f"Best F1 score: {best_f1:.3f} at epoch {best_epoch}")
    log(f"Model saved to: {CLASSIFIER_PATH}")
    log(f"Training log: {log_file}")
    log(f"History JSON: {history_file}")
    
    # Final confusion matrix
    best_cm = history[best_epoch - 1]['confusion_matrix']
    log(f"\nBest Epoch Confusion Matrix:")
    log(f"                Predicted")
    log(f"              Neg    Pos")
    log(f"Actual  Neg  {best_cm[0][0]:5d}  {best_cm[0][1]:5d}")
    log(f"        Pos  {best_cm[1][0]:5d}  {best_cm[1][1]:5d}")
    
    return history


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Train improved ResNet18 classifier")
    p.add_argument('--train', type=Path, default=Path('data/train.csv'))
    p.add_argument('--val', type=Path, default=Path('data/val.csv'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--input-size', type=int, default=224, help="Input image size (default: 224)")
    p.add_argument('--focal-loss', action='store_true', help="Use focal loss instead of CE")
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--no-balancing', action='store_true', help="Disable class balancing")
    p.add_argument('--early-stop', type=int, default=7, help="Early stopping patience")
    args = p.parse_args()
    
    train(
        args.train,
        args.val,
        args.device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.input_size,
        args.focal_loss,
        args.focal_gamma,
        not args.no_balancing,
        args.early_stop
    )
