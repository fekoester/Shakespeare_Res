"""
Train character-level Transformer on Shakespeare.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler

from utils.data_utils import load_corpus, build_vocab, TextDataset
from utils.training_utils import set_seed, get_hash, count_parameters, run_epoch
from models.transformer import CharTransformer

# Paths & outputs
DATA_PATH   = "data/shakespeare.txt"
RESULTS_CSV = "results_transformer.csv"
CKPT_DIR    = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# Base hyperparameters
BASE = {
    "TRAIN_SPLIT": 0.9,
    "BATCH":       1024,
    "EPOCHS":      5,
    "LR":          1e-4,
    "SEQ":         32,
    "EMB":         16,
}

# Transformer sweep configurations
TRANSFORMER_CONFIGS = [
    {"HID": 64,  "HEADS": 4,  "LAYERS": 4},
    {"HID": 72,  "HEADS": 8,  "LAYERS": 8},
    {"HID": 128, "HEADS": 8,  "LAYERS": 8},
    {"HID": 356, "HEADS": 8,  "LAYERS": 8},
    {"HID": 256, "HEADS": 16, "LAYERS": 16},
]

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load & split
    text = load_corpus(DATA_PATH)
    vocab, c2i, _ = build_vocab(text)
    split = int(len(text) * BASE["TRAIN_SPLIT"])
    train_ds = TextDataset(text[:split], BASE["SEQ"], c2i)
    test_ds  = TextDataset(text[split:], BASE["SEQ"], c2i)

    train_dl = DataLoader(train_ds, BASE["BATCH"], shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  BASE["BATCH"], shuffle=False, num_workers=2)

    all_results = []

    for cfg in TRANSFORMER_CONFIGS:
        h = dict(BASE, HID=cfg["HID"], HEADS=cfg["HEADS"], LAYERS=cfg["LAYERS"])
        name = f"trans_{get_hash(h)}"
        print(f"\n=== Training {name} ===")

        model = CharTransformer(
            vocab_size=len(vocab),
            embed_size=h["EMB"],
            num_layers=h["LAYERS"],
            num_heads=h["HEADS"],
            hidden_size=h["HID"],
            seq_len=h["SEQ"]
        ).to(device)
        print("Trainable parameters:", count_parameters(model))

        optimizer = optim.Adam(model.parameters(), lr=h["LR"])
        criterion = nn.CrossEntropyLoss()
        scaler    = GradScaler()

        for epoch in range(1, h["EPOCHS"] + 1):
            train_loss = run_epoch(model, train_dl,  criterion, optimizer, scaler, device)
            test_loss  = run_epoch(model, test_dl,   criterion, None,      None,   device)
            print(f"[{name}] Epoch {epoch}: train {train_loss:.4f} | test {test_loss:.4f}")

            all_results.append({
                "model":      name,
                "epoch":      epoch,
                "train_loss": train_loss,
                "test_loss":  test_loss,
                **h
            })

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pth"))

    # Write CSV
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

if __name__ == "__main__":
    main()

