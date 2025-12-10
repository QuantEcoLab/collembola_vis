from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from .config import CLASSIFIER_PATH, CLASSIFIER_THRESHOLD


def get_val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_classifier(device: str | torch.device = 'cpu'):
    model = models.resnet18(weights=None)
    # Expect two-class head
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)
    if CLASSIFIER_PATH.exists():
        state = torch.load(CLASSIFIER_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval().to(device)
    return model


def predict_proba(img: Image.Image, model, transform=None, device='cpu') -> float:
    if transform is None:
        transform = get_val_transform()
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        p = probs[0, 1].item()
    return p


def is_organism(img: Image.Image, model, threshold: float = CLASSIFIER_THRESHOLD, transform=None, device='cpu') -> Tuple[bool, float]:
    p = predict_proba(img, model, transform, device)
    return (p >= threshold), p
