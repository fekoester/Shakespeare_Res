"""
Shared helpers: seeding, parameter counting, training/eval loops.
"""

import math, random, hashlib, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

__all__ = ["set_seed", "get_hash", "count_parameters", "run_epoch"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_hash(hparams: dict, length=8) -> str:
    pairs = "&".join(f"{k}={v}" for k, v in sorted(hparams.items()))
    return hashlib.sha1(pairs.encode()).hexdigest()[:length]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(
    model, dataloader, criterion, optimizer=None, scaler=None, device="cpu"
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                logits = model(X)
                loss = criterion(logits, y)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / len(dataloader)

