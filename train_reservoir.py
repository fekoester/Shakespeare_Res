"""
Train classic reservoir network on Shakespeare.
"""

import os, torch, pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler

from utils.data_utils import load_corpus, build_vocab, TextDataset
from utils.training_utils import set_seed, get_hash, count_parameters, run_epoch
from models.reservoir import ReservoirNet

DATA_PATH = "data/shakespeare.txt"
RESULTS_CSV = "results_reservoir.csv"
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)

h = {
    "TRAIN_SPLIT": 0.9, "BATCH": 1024, "EPOCHS": 5,
    "LR": 1e-4, "SEQ": 32, "EMB": 16, "RES": 750
}

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(DATA_PATH)
    vocab, c2i, _ = build_vocab(text)
    split = int(len(text) * h["TRAIN_SPLIT"])
    train_ds = TextDataset(text[:split], h["SEQ"], c2i)
    test_ds  = TextDataset(text[split:], h["SEQ"], c2i)

    train_dl = DataLoader(train_ds, h["BATCH"], shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  h["BATCH"], shuffle=False, num_workers=2)

    model = ReservoirNet(len(vocab), h["EMB"], h["RES"]).to(device)
    print("Params:", count_parameters(model))
    opt  = optim.Adam(model.classifier.parameters(), lr=h["LR"])
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler()

    results = []
    for epoch in range(1, h["EPOCHS"] + 1):
        tr = run_epoch(model, train_dl, crit, opt, scaler, device)
        te = run_epoch(model, test_dl,  crit, None, None,  device)
        print(f"Epoch {epoch}: train {tr:.4f} | test {te:.4f}")
        results.append({"epoch": epoch, "train_loss": tr, "test_loss": te})

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"reservoir_{get_hash(h)}.pth"))

if __name__ == "__main__":
    main()

