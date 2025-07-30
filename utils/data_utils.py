"""
Utilities for loading the Shakespeare corpus and building PyTorch datasets.
"""

import os
import torch
from torch.utils.data import Dataset

def load_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

def build_vocab(text: str):
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return vocab, char2idx, idx2char

class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int, char2idx: dict):
        self.text = text
        self.seq_len = seq_len
        self.char2idx = char2idx

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.text[idx : idx + self.seq_len]
        y_char = self.text[idx + self.seq_len]
        x_enc = [self.char2idx[c] for c in x_seq]
        y_enc = self.char2idx[y_char]
        return torch.tensor(x_enc, dtype=torch.long), torch.tensor(y_enc, dtype=torch.long)

