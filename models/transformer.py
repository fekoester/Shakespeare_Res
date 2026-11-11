"""
Lightweight character-level Transformer for comparison.

Defaults match v0.1 (and the paper): PyTorch's post-LayerNorm encoder layer with dropout 0.1.
Post-LN stacks are fragile early in training; at learning rates well above the paper's 1e-4 the deeper
configurations can get stuck predicting only the character frequencies. norm_first=True (pre-LN) and
a learning-rate warmup (train.py --warmup_steps) are the standard remedies.
"""

import math

import torch
import torch.nn as nn

__all__ = ["CharTransformer"]

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_size, seq_len, dropout=0.1,
                 norm_first=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.register_buffer("pos", self._positional_encoding(seq_len, embed_size), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout=dropout,
                                               batch_first=True, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(embed_size, vocab_size)

    def _positional_encoding(self, T, D):
        P = torch.zeros(T, D)
        pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        P[:, 0::2] = torch.sin(pos * div)
        P[:, 1::2] = torch.cos(pos * div)
        return P

    def forward(self, x):
        B, T = x.shape
        e = self.embed(x) + self.pos[:T, :]
        h = self.transformer(e)
        return self.fc(h[:, -1, :])  # last position

