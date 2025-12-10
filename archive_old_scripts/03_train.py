from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

from collembola_pipeline.config import MODELS_DIR, CLASSIFIER_PATH


class CropDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool = False, size: int = 128):
        self.df = pd.read_csv(csv_path)
        if train:
            self.t = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.t = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row['img_path'])
        label = int(row['label'])
        img = Image.open(path).convert('RGB')
        return self.t(img), label


def train(train_csv: Path, val_csv: Path, device='cpu', epochs: int = 10, batch_size: int = 64, lr: float = 1e-4):
    ds_tr = CropDataset(train_csv, train=True)
    ds_va = CropDataset(val_csv, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        train_loss = running / total
        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CLASSIFIER_PATH)
            print(f"Saved best model to {CLASSIFIER_PATH}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, default=Path('data/train.csv'))
    p.add_argument('--val', type=Path, default=Path('data/val.csv'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()

    train(args.train, args.val, args.device, args.epochs, args.batch_size, args.lr)
