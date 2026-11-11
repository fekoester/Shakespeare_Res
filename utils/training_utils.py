"""
Shared helpers: seeding, parameter counting, training/eval loops.
"""

import hashlib
import random
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

__all__ = ["set_seed", "get_hash", "count_parameters", "amp_context", "run_epoch"]

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
    """Trainable parameters (frozen reservoir, embedding and LayerNorm are excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def amp_context(device):
    """bf16 autocast on CUDA only (on CPU it is slower and less accurate)."""
    dev = torch.device(device)
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext()

def run_epoch(
    model, dataloader, criterion, optimizer=None, scaler=None, device="cpu"
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()

            with amp_context(device):
                logits = model(X)
                loss = criterion(logits.float(), y)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)

