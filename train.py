#!/usr/bin/env python3
"""
Train a character-level model on Shakespeare and report bits per character (bpc).

    python train.py --model reservoir --res 750
    python train.py --model reservoir --res 750 --readout ridge      # closed-form ESN readout, one pass
    python train.py --model att_reservoir --res 150 --hid 25
    python train.py --model transformer --hid 128 --heads 8 --layers 8

Splits (contiguous): train | validation (--val_frac before the test split) | test (last --test_frac;
0.1 as in v0.1). The epoch with the best validation loss is kept and evaluated once on the test split
(with --val_frac 0 the last epoch is used).

--shards N --shard_epochs K reproduces the paper's schedule: the training text is cut into N contiguous
shards and each shard is trained for K passes before moving to the next; one pass over all shards is an
epoch. `--protocol paper` sets the paper's full setup (Köster & Uchida, Phys. Rev. Applied 26, 024051):

    python train.py --protocol paper --model transformer --hid 256 --heads 16 --layers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.att_reservoir import AttReservoirNet
from models.reservoir import ReservoirNet
from models.transformer import CharTransformer
from utils.data_utils import build_vocab, encode, iterate_windows, load_corpus, shard_tokens, split_tokens
from utils.training_utils import amp_context, count_parameters, get_hash, set_seed

LN2 = math.log(2.0)

# The setup of the paper: Adam 1e-4, batch 1024, 6 contiguous shards (5 train, 1 test), 5 passes per shard,
# no validation split, all windows evaluated.
PROTOCOLS = {
    "paper": dict(lr=1e-4, batch=1024, test_frac=1 / 6, val_frac=0.0, shards=5, shard_epochs=5, epochs=1,
                  steps_per_epoch=0, eval_windows=0, test_windows=0, warmup_steps=0),
}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="att_reservoir", choices=["reservoir", "att_reservoir", "transformer"])
    ap.add_argument("--data", default="data/shakespeare.txt")
    ap.add_argument("--no_lowercase", action="store_true", help="keep case (v0.1 lowercases the corpus)")
    ap.add_argument("--seq", type=int, default=32, help="context window in characters")
    ap.add_argument("--emb", type=int, default=16)
    # reservoir models
    ap.add_argument("--res", type=int, default=150, help="reservoir size")
    ap.add_argument("--hid", type=int, default=25,
                    help="att_reservoir: generated readout size; transformer: feed-forward size")
    ap.add_argument("--spectral_radius", type=float, default=0.9)
    ap.add_argument("--input_scale", type=float, default=1.0)
    ap.add_argument("--legacy_scaling", action="store_true",
                    help="v0.1 reservoir scaling (spectral radius ~1.1); reproduces old checkpoints")
    ap.add_argument("--leak_min", type=float, default=1.0,
                    help="< 1: multi-timescale reservoir, leak rates log-spaced in [leak_min, 1] (as in LAERC)")
    ap.add_argument("--readout", default="adam", choices=["adam", "ridge"], help="reservoir model only")
    ap.add_argument("--ridge_windows", type=int, default=1_000_000, help="training windows for the ridge fit")
    # transformer
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1, help="transformer dropout (v0.1: 0.1)")
    ap.add_argument("--norm_first", action="store_true", help="pre-LN transformer layers (v0.1: post-LN)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="0: one pass over all training windows")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=0, help="linear learning-rate warmup")
    ap.add_argument("--test_frac", type=float, default=0.1, help="last fraction of the corpus used for testing")
    ap.add_argument("--val_frac", type=float, default=0.05, help="validation fraction before the test split (0: none)")
    ap.add_argument("--shards", type=int, default=1, help="contiguous training shards (paper: 5)")
    ap.add_argument("--shard_epochs", type=int, default=1, help="passes over each shard before the next (paper: 5)")
    ap.add_argument("--protocol", default="", choices=["", "paper"],
                    help="preset: 'paper' = the setup of the paper (overrides lr, batch, splits, schedule)")
    ap.add_argument("--eval_windows", type=int, default=100_000, help="validation windows per evaluation (0: all)")
    ap.add_argument("--test_windows", type=int, default=0, help="test windows for the final evaluation (0: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--name", default="", help="run name (default: model + hash of the config)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def build_model(cfg, vocab_size: int) -> torch.nn.Module:
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    rkw = dict(spectral_radius_target=get("spectral_radius", 0.9), input_scale=get("input_scale", 1.0),
               legacy_scaling=get("legacy_scaling", False), leak_min=get("leak_min", 1.0) or 1.0)
    m = get("model")
    if m == "reservoir":
        return ReservoirNet(vocab_size, get("emb"), get("res"), **rkw)
    if m == "att_reservoir":
        return AttReservoirNet(vocab_size, get("emb"), get("res"), get("hid"), **rkw)
    if m == "transformer":
        return CharTransformer(vocab_size, get("emb"), get("layers"), get("heads"), get("hid"), get("seq"),
                               dropout=get("dropout", 0.1), norm_first=get("norm_first", False))
    raise ValueError(m)


def trainable_module(model):
    """The part that is optimised (the readout for reservoir models, everything for the transformer)."""
    if isinstance(model, ReservoirNet):
        return model.classifier
    if isinstance(model, AttReservoirNet):
        return model.head
    return model


@torch.no_grad()
def evaluate(model, tokens, cfg, device, max_windows=0):
    """Returns (cross-entropy in nats, accuracy) over the windows of `tokens`."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in iterate_windows(tokens, cfg.seq, 4096, max_windows=max_windows):
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    return loss_sum / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def fit_ridge(model: ReservoirNet, train_tok, val_tok, cfg, device, lambdas=None):
    """
    Echo-state-network readout in closed form: accumulate Phi^T Phi and Phi^T Y over training windows
    (Phi = reservoir features + bias, Y = one-hot next character), solve the ridge regression for a grid
    of regularisers via one eigendecomposition, and pick the regulariser and a softmax temperature tau
    (logits = tau * Phi W) by validation cross-entropy. The result is written into model.classifier.
    """
    model.eval()
    V = model.classifier.out_features
    Rn = model.classifier.in_features
    A = torch.zeros(Rn + 1, Rn + 1, dtype=torch.float64, device=device)
    B = torch.zeros(Rn + 1, V, dtype=torch.float64, device=device)
    g = torch.Generator().manual_seed(cfg.seed)
    n = 0
    it = iterate_windows(train_tok, cfg.seq, 8192, shuffle=True, generator=g, max_windows=cfg.ridge_windows)
    for x, y in tqdm(it, desc="ridge: accumulate", disable=cfg.quiet, leave=False):
        phi = model.extractor(x.to(device)).double()
        phi = torch.cat([phi, torch.ones(len(phi), 1, dtype=phi.dtype, device=device)], 1)
        A += phi.T @ phi
        B.index_add_(1, y.to(device), phi.T)
        n += len(phi)
    evals, Q = torch.linalg.eigh(A / n)
    QtB = Q.T @ (B / n)

    xv, yv = next(iterate_windows(val_tok, cfg.seq, 50_000, max_windows=50_000))
    phiv = model.extractor(xv.to(device)).double()
    phiv = torch.cat([phiv, torch.ones(len(phiv), 1, dtype=phiv.dtype, device=device)], 1)
    yv = yv.to(device)
    best = (math.inf, None, None, None)
    for lam in lambdas or [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        W = Q @ (QtB / (evals + lam).unsqueeze(1))
        out = phiv @ W
        for tau in torch.logspace(-1, 2.5, 36).tolist():
            ce = F.cross_entropy((tau * out).float(), yv).item()
            if ce < best[0]:
                best = (ce, lam, tau, W)
    ce, lam, tau, W = best
    model.classifier.weight.copy_((tau * W[:-1]).T.float())
    model.classifier.bias.copy_((tau * W[-1]).float())
    return {"ridge_lambda": lam, "ridge_tau": tau, "ridge_windows": n, "ridge_val_ce": ce}


def run(cfg, verbose=True):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    text = load_corpus(cfg.data) if not cfg.no_lowercase else open(cfg.data, encoding="utf-8").read()
    vocab, c2i, _ = build_vocab(text)
    tokens = encode(text, c2i)
    train_tok, val_tok, test_tok = split_tokens(tokens, 1.0 - cfg.test_frac, cfg.val_frac)
    has_val = len(val_tok) > cfg.seq

    model = build_model(cfg, len(vocab)).to(device)
    if cfg.readout == "ridge" and cfg.model != "reservoir":
        raise ValueError("--readout ridge is only defined for the linear readout of --model reservoir")
    if cfg.readout == "ridge" and not has_val:
        raise ValueError("--readout ridge chooses its regulariser on the validation split; use --val_frac > 0")
    n_params = count_parameters(model)
    name = cfg.name or f"{cfg.model}_{get_hash({k: v for k, v in vars(cfg).items() if k not in ('quiet', 'device')})}"
    out_dir = os.path.join(cfg.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[{name}] {cfg.model}: {n_params:,} trainable parameters, vocab {len(vocab)}, "
              f"{len(train_tok):,}/{len(val_tok):,}/{len(test_tok):,} train/val/test characters")

    history, extra = [], {}
    t0 = time.time()
    best = {"val_loss": math.inf}
    if cfg.model == "reservoir" and cfg.readout == "ridge":
        extra = fit_ridge(model, train_tok, val_tok, cfg, device)
        v_loss, v_acc = evaluate(model, val_tok, cfg, device, cfg.eval_windows)
        history.append({"epoch": 0, "val_loss": v_loss, "val_bpc": v_loss / LN2, "val_acc": v_acc,
                        "time_s": time.time() - t0})
        best = {"val_loss": v_loss, "epoch": 0, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    else:
        opt = torch.optim.Adam(trainable_module(model).parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: min(1.0, (step + 1) / cfg.warmup_steps) if cfg.warmup_steps > 0 else 1.0)
        g = torch.Generator().manual_seed(cfg.seed)
        max_w = cfg.steps_per_epoch * cfg.batch if cfg.steps_per_epoch else 0
        shards = shard_tokens(train_tok, cfg.shards)
        for ep in range(1, cfg.epochs + 1):
            model.train()
            te = time.time()
            run_loss, run_n, shard_losses = 0.0, 0, []
            for si, shard in enumerate(shards):
                for k in range(cfg.shard_epochs):
                    s_loss, s_n = 0.0, 0
                    it = iterate_windows(shard, cfg.seq, cfg.batch, shuffle=True, generator=g, max_windows=max_w)
                    total = (max_w or (len(shard) - cfg.seq)) // cfg.batch
                    desc = f"epoch {ep}" + (f" shard {si + 1}/{len(shards)} pass {k + 1}" if len(shards) > 1 else "")
                    for x, y in tqdm(it, total=total, desc=desc, disable=not verbose or cfg.quiet, leave=False):
                        x, y = x.to(device), y.to(device)
                        with amp_context(device):
                            loss = F.cross_entropy(model(x).float(), y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        sched.step()
                        s_loss += loss.item() * len(y)
                        s_n += len(y)
                    if len(shards) > 1 or cfg.shard_epochs > 1:
                        shard_losses.append({"shard": si + 1, "pass": k + 1, "train_loss": s_loss / max(s_n, 1)})
                    run_loss += s_loss
                    run_n += s_n
            row = {"epoch": ep, "train_loss": run_loss / max(run_n, 1), "time_s": time.time() - te}
            if shard_losses:
                row["shards"] = shard_losses
            if has_val:
                v_loss, v_acc = evaluate(model, val_tok, cfg, device, cfg.eval_windows)
                row.update({"val_loss": v_loss, "val_bpc": v_loss / LN2, "val_acc": v_acc})
            history.append(row)
            # best epoch by validation loss; without a validation split, the last epoch
            score = row["val_loss"] if has_val else -ep
            if score < best["val_loss"]:
                best = {"val_loss": score, "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if verbose:
                val = f" | val {row['val_bpc']:.3f} bpc, acc {row['val_acc'] * 100:.1f}%" if has_val else ""
                print(f"[{name}] epoch {ep}: train {row['train_loss'] / LN2:.3f} bpc{val} ({row['time_s']:.0f}s)",
                      flush=True)

    model.load_state_dict(best["state"])
    t_loss, t_acc = evaluate(model, test_tok, cfg, device, cfg.test_windows)
    metrics = {"name": name, "model": cfg.model, "trainable_params": n_params, "best_epoch": best["epoch"],
               "val_bpc": best["val_loss"] / LN2 if has_val else None, "test_loss": t_loss, "test_bpc": t_loss / LN2,
               "test_acc": t_acc, "train_time_s": time.time() - t0, "history": history, **extra,
               "config": {k: v for k, v in vars(cfg).items() if k != "quiet"}}
    torch.save({"model": model.state_dict(), "config": metrics["config"], "vocab": vocab},
               os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"[{name}] test: {metrics['test_bpc']:.3f} bpc, accuracy {t_acc * 100:.1f}% "
              f"(best epoch {best['epoch']})")
    return metrics, model


def apply_protocol(cfg, argv):
    """Fill in a protocol preset; flags given explicitly on the command line still win."""
    if cfg.protocol:
        given = {a.split("=")[0].lstrip("-") for a in (argv or []) if a.startswith("--")}
        for k, v in PROTOCOLS[cfg.protocol].items():
            if k not in given:
                setattr(cfg, k, v)
    return cfg


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    return run(apply_protocol(build_parser().parse_args(argv), argv))


if __name__ == "__main__":
    main()
