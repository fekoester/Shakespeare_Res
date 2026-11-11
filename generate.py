#!/usr/bin/env python3
"""
Generate text from a trained model, one character at a time from the last --seq characters.

    python generate.py checkpoints/<run>/model.pt --prompt "to be, or not to be" --length 400
"""

import argparse

import torch

from train import build_model


@torch.no_grad()
def generate(model, vocab, prompt, length, seq, temperature=0.8, top_k=0, generator=None, device="cpu"):
    c2i = {c: i for i, c in enumerate(vocab)}
    ids = [c2i[c] for c in prompt if c in c2i] or [c2i.get(" ", 0)]
    model.eval()
    for _ in range(length):
        x = torch.tensor([ids[-seq:]], device=device)
        logits = model(x)[0].float() / max(temperature, 1e-4)
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, len(logits))).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        ids.append(int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)))
    return "".join(vocab[i] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="to be, or not to be")
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg, vocab = ck["config"], ck["vocab"]
    model = build_model(cfg, len(vocab))
    model.load_state_dict(ck["model"])
    if not cfg.get("no_lowercase"):
        a.prompt = a.prompt.lower()
    g = torch.Generator().manual_seed(a.seed)
    print(generate(model, vocab, a.prompt, a.length, cfg["seq"], a.temperature, a.top_k, g))


if __name__ == "__main__":
    main()
