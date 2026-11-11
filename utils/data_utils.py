"""
Utilities for loading the Shakespeare corpus and building PyTorch datasets.
"""

import torch
from torch.utils.data import Dataset


def load_corpus(path: str) -> str:
    with open(path, encoding="utf-8") as f:
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



# --------------------------------------------------------------------------- fast window access
def encode(text: str, char2idx: dict) -> torch.Tensor:
    """Encode the whole corpus once (uint8 is enough for a character vocabulary)."""
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)


def split_tokens(tokens: torch.Tensor, train_frac: float = 0.9, val_frac: float = 0.05):
    """
    Contiguous splits. The test split is the last (1 - train_frac) of the corpus, as in v0.1; the
    validation split is the last val_frac of the corpus before it (taken out of the old training part).
    """
    n = len(tokens)
    test_start = int(n * train_frac)
    val_start = int(n * (train_frac - val_frac))
    return tokens[:val_start], tokens[val_start:test_start], tokens[test_start:]


def window_batch(tokens: torch.Tensor, starts: torch.Tensor, seq_len: int):
    """Windows tokens[s:s+seq_len] and their next characters tokens[s+seq_len]."""
    idx = starts.unsqueeze(1) + torch.arange(seq_len)
    return tokens[idx], tokens[starts + seq_len]


def iterate_windows(tokens, seq_len, batch_size, shuffle=False, generator=None, max_windows=0):
    """
    Batches of (x [B, seq_len], y [B]) over all windows of `tokens` (or `max_windows` of them: random
    ones when shuffling, evenly spaced otherwise, so evaluation always uses the same windows).
    """
    n = len(tokens) - seq_len
    if shuffle:
        starts = torch.randperm(n, generator=generator)
        if max_windows:
            starts = starts[:max_windows]
    elif max_windows and max_windows < n:
        starts = torch.linspace(0, n - 1, max_windows).long()
    else:
        starts = torch.arange(n)
    for i in range(0, len(starts), batch_size):
        yield window_batch(tokens, starts[i:i + batch_size], seq_len)
