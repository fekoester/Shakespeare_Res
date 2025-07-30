"""
Neural-net-generated attention weights on top of a reservoir.
"""

import torch, torch.nn as nn
from .reservoir import _ReservoirExtractor

__all__ = ["AttReservoirNet"]

class _DynAttentionHead(nn.Module):
    def __init__(self, reservoir_size, hidden_size, vocab_size):
        super().__init__()
        self.generate = nn.Sequential(
            nn.Linear(reservoir_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * reservoir_size)
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.reservoir_size = reservoir_size

    def forward(self, r_state):
        B = r_state.size(0)
        W = self.generate(r_state).view(B, self.hidden_size, self.reservoir_size)
        out = (W @ r_state.unsqueeze(-1)).squeeze(-1)
        return self.classifier(out)

class AttReservoirNet(nn.Module):
    def __init__(self, vocab_size, embed_size, reservoir_size, hidden_size):
        super().__init__()
        self.extractor = _ReservoirExtractor(vocab_size, embed_size, reservoir_size)
        self.head = _DynAttentionHead(reservoir_size, hidden_size, vocab_size)

    def forward(self, x):
        r = self.extractor(x)
        return self.head(r)

