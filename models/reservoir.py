"""
Classic (static) reservoir feature extractor + linear classifier.
"""

import torch, torch.nn as nn
import math

__all__ = ["ReservoirNet"]

class _ReservoirExtractor(nn.Module):
    def __init__(self, vocab_size, embed_size, reservoir_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = False

        self.R = nn.Parameter(
            torch.randn(reservoir_size, reservoir_size) / math.sqrt(reservoir_size) / 0.9,
            requires_grad=False
        )
        self.Win = nn.Parameter(
            torch.randn(embed_size, reservoir_size) / math.sqrt(embed_size) / 0.9,
            requires_grad=False
        )
        self.norm = nn.LayerNorm(reservoir_size)

    def forward(self, x):
        # x: [B, T]
        x = self.embed(x)
        B, T, _ = x.size()
        state = torch.zeros(B, self.R.size(0), device=x.device)
        for t in range(T):
            state = torch.tanh(state @ self.R + x[:, t, :] @ self.Win)
        return self.norm(state)

class ReservoirNet(nn.Module):
    def __init__(self, vocab_size, embed_size, reservoir_size):
        super().__init__()
        self.extractor = _ReservoirExtractor(vocab_size, embed_size, reservoir_size)
        self.classifier = nn.Linear(reservoir_size, vocab_size)

    def forward(self, x):
        r = self.extractor(x)
        return self.classifier(r)

