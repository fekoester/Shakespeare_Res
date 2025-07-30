"""
Lightweight character-level Transformer for comparison.
"""

import math, torch, torch.nn as nn

__all__ = ["CharTransformer"]

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_size, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos = self._positional_encoding(seq_len, embed_size)
        enc_layer = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
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
        e = self.embed(x) + self.pos[:T, :].to(x.device)
        h = self.transformer(e)
        return self.fc(h[:, -1, :])  # last position

