"""
Classic (static) reservoir feature extractor + linear classifier.

The extractor embeds a character window with a fixed random embedding, runs a fixed random tanh
reservoir over it from a zero state, and returns the layer-normalised final state:

    s_t = tanh(s_{t-1} R + e_t W_in),   features = LN(s_T)

With leak_min < 1 the reservoir has a spectrum of timescales (as in LAERC): unit i integrates with its
own leak rate a_i, log-spaced in [leak_min, 1],

    s_t = s_{t-1} + a * (tanh(s_{t-1} R + e_t W_in) - s_{t-1}),

so its memory ranges from ~1 to ~1/leak_min characters. leak_min = 1 is the plain reservoir above.

Nothing in the extractor is trained (the LayerNorm has no trainable affine either).
"""

import math

import torch
import torch.nn as nn

__all__ = ["ReservoirNet", "spectral_radius"]


@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    return torch.linalg.eigvals(W.detach().double().cpu()).abs().max().item()


class _ReservoirExtractor(nn.Module):
    def __init__(self, vocab_size, embed_size, reservoir_size, spectral_radius_target=0.9, input_scale=1.0,
                 legacy_scaling=False, leak_min=1.0):
        """
        legacy_scaling=True reproduces v0.1 exactly: R = randn/sqrt(N)/0.9 and W_in = randn/sqrt(E)/0.9,
        which gives a spectral radius of about 1.1 (the 0.9 divides instead of multiplying).
        Otherwise R is scaled to the requested spectral radius and W_in to input_scale/sqrt(E).
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = False

        R = torch.randn(reservoir_size, reservoir_size) / math.sqrt(reservoir_size)
        Win = torch.randn(embed_size, reservoir_size) / math.sqrt(embed_size)
        if legacy_scaling:
            R, Win = R / 0.9, Win / 0.9
        else:
            R = R * (spectral_radius_target / spectral_radius(R))
            Win = Win * input_scale
        self.R = nn.Parameter(R, requires_grad=False)
        self.Win = nn.Parameter(Win, requires_grad=False)
        if not 0.0 < leak_min <= 1.0:
            raise ValueError("leak_min must be in (0, 1]")
        self.leaky = leak_min < 1.0
        if self.leaky:
            self.register_buffer("leak", torch.logspace(math.log10(leak_min), 0.0, reservoir_size))
        self.norm = nn.LayerNorm(reservoir_size)
        for p in self.norm.parameters():  # never optimised in any script; keep it out of parameter counts
            p.requires_grad = False

    def forward(self, x):
        # x: [B, T] character ids
        u = self.embed(x) @ self.Win  # [B, T, N] input drive for all steps at once
        state = torch.zeros(x.size(0), self.R.size(0), device=x.device, dtype=u.dtype)
        if self.leaky:
            a = self.leak.to(u.dtype)
            for t in range(x.size(1)):
                state = state + a * (torch.tanh(state @ self.R + u[:, t]) - state)
        else:
            for t in range(x.size(1)):
                state = torch.tanh(state @ self.R + u[:, t])
        return self.norm(state)


class ReservoirNet(nn.Module):
    def __init__(self, vocab_size, embed_size, reservoir_size, **reservoir_kw):
        super().__init__()
        self.extractor = _ReservoirExtractor(vocab_size, embed_size, reservoir_size, **reservoir_kw)
        self.classifier = nn.Linear(reservoir_size, vocab_size)

    def forward(self, x):
        return self.classifier(self.extractor(x))
