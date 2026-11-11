#!/usr/bin/env python3
"""
Bits per character vs trainable parameters for the three model families on Shakespeare (the v0.1 /
paper sweep configurations).

  --protocol paper   the setup of the paper (Adam 1e-4, batch 1024, 5 train shards x 5 passes, last
                     sixth for testing). This is the comparison to run; it needs a GPU for the large
                     Transformers (the paper used an RTX 4090).
  --protocol quick   a short CPU sanity check (2,000 steps x 256 windows, lr 3e-3). Far from converged,
                     and the deeper post-LN Transformers do not train at this learning rate.

    python benchmark.py --protocol paper --out results/paper
    python benchmark.py --protocol quick --out results
"""

import argparse
import json
import os

import train

RUNS = [
    ("reservoir", "adam", ["--model", "reservoir", "--res", "750"]),
    ("reservoir", "ridge", ["--model", "reservoir", "--res", "750", "--readout", "ridge"]),
    ("reservoir", "adam, v0.1 scaling", ["--model", "reservoir", "--res", "750", "--legacy_scaling"]),
    *[("att_reservoir", "", ["--model", "att_reservoir", "--res", str(r), "--hid", str(h)])
      for r, h in [(75, 13), (75, 19), (100, 20), (150, 25), (160, 30)]],
    ("att_reservoir", "v0.1 scaling", ["--model", "att_reservoir", "--res", "150", "--hid", "25", "--legacy_scaling"]),
    ("reservoir", "multi-timescale", ["--model", "reservoir", "--res", "750", "--leak_min", "0.1"]),
    ("att_reservoir", "multi-timescale", ["--model", "att_reservoir", "--res", "150", "--hid", "25",
                                          "--leak_min", "0.1"]),
    *[("transformer", "", ["--model", "transformer", "--hid", str(f), "--heads", str(h), "--layers", str(L)])
      for f, h, L in [(64, 4, 4), (72, 8, 8), (128, 8, 8), (356, 8, 8), (256, 16, 16)]],
    # the deeper post-LN Transformers are unstable at the shared lr; also run them at a lower one
    *[("transformer", "lr 1e-3", ["--model", "transformer", "--hid", str(f), "--heads", str(h), "--layers", str(L),
                                  "--lr", "1e-3"])
      for f, h, L in [(128, 8, 8), (356, 8, 8), (256, 16, 16)]],
]
FAMILY = {"reservoir": "Classic reservoir", "att_reservoir": "Attention-enhanced reservoir",
          "transformer": "Transformer"}


def describe(args):
    d = dict(zip(args[::2], args[1::2]))  # (the lr variant is reported in the variant column)
    m = d["--model"]
    if m == "reservoir":
        return f"R={d['--res']}"
    if m == "att_reservoir":
        return f"R={d['--res']}, H={d['--hid']}"
    return f"L={d['--layers']}, heads={d['--heads']}, ffn={d['--hid']}"


def radius_sweep(a, common):
    """Attention-enhanced reservoir (R=150, H=25) at several spectral radii."""
    path = os.path.join(a.out, "radius_sweep.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for rho in a.radii:
        if str(rho) in res:
            continue
        print(f"\n=== radius sweep: rho={rho} ===", flush=True)
        m, _ = train.main(["--model", "att_reservoir", "--res", "150", "--hid", "25", "--spectral_radius", str(rho)]
                          + common)
        res[str(rho)] = {"test_bpc": m["test_bpc"], "test_acc": m["test_acc"]}
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = sorted(float(k) for k in res)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(rhos, [res[str(r)]["test_bpc"] for r in rhos], marker="o")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("spectral radius of the reservoir")
    ax.set_ylabel("test bits per character")
    ax.set_title("Attention-enhanced reservoir, R=150, H=25")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "radius_sweep.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="paper", choices=["paper", "quick"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_windows", type=int, default=50_000)
    ap.add_argument("--test_windows", type=int, default=200_000)
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--radii", type=float, nargs="*", default=[0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6],
                    help="spectral radii for the attention-reservoir sweep (empty: skip)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "benchmark.json")
    results = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["family"], r["variant"], r["config"]) for r in results}
    if a.protocol == "paper":
        common = ["--protocol", "paper", "--out_dir", os.path.join(a.out, "runs"), "--quiet"]
    else:
        common = ["--epochs", "1", "--steps_per_epoch", str(a.steps), "--batch", str(a.batch), "--lr", str(a.lr),
                  "--eval_windows", str(a.eval_windows), "--test_windows", str(a.test_windows),
                  "--out_dir", os.path.join(a.out, "runs"), "--quiet"]
    for fam, variant, args in RUNS:
        if a.protocol == "paper" and variant == "lr 1e-3":
            continue  # learning-rate rescue runs only make sense for the quick check
        cfg = describe(args)
        if (fam, variant, cfg) in done:
            continue
        print(f"\n=== {FAMILY[fam]} {cfg} {variant} ===", flush=True)
        extra = ["--val_frac", "0.05"] if "ridge" in args and a.protocol == "paper" else []
        m, _ = train.main(common + args + extra)  # run-specific arguments win
        results.append({"family": fam, "variant": variant, "config": cfg, "trainable_params": m["trainable_params"],
                        "test_bpc": m["test_bpc"], "test_loss_nats": m["test_loss"], "test_acc": m["test_acc"],
                        "val_bpc": m["val_bpc"],
                        "train_time_s": m["train_time_s"], "run": m["name"]})
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    lines = ["| model | config | trainable params | test bpc | test accuracy | train time |", "|---|---|---|---|---|---|"]
    for r in results:
        label = FAMILY[r["family"]] + (f" ({r['variant']})" if r["variant"] else "")
        lines.append(f"| {label} | {r['config']} | {r['trainable_params']:,} | {r['test_bpc']:.3f} | "
                     f"{r['test_acc'] * 100:.1f}% | {r['train_time_s'] / 60:.1f} min |")
    md = "\n".join(lines)
    open(os.path.join(a.out, "benchmark.md"), "w").write(md + "\n")
    print(md)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    markers = {"reservoir": "s", "att_reservoir": "o", "transformer": "^"}
    LR_VARIANTS = ("", "adam", "lr 1e-3")  # same model, different learning rate: plot the best one
    DIVERGED = 4.0  # bpc above this = collapsed to the character frequencies
    top = 3.45
    for fam in FAMILY:
        by_cfg = {}
        for r in results:
            if r["family"] == fam and r["variant"] in LR_VARIANTS:
                if r["config"] not in by_cfg or r["test_bpc"] < by_cfg[r["config"]]["test_bpc"]:
                    by_cfg[r["config"]] = r
        pts = sorted(by_cfg.values(), key=lambda r: r["trainable_params"])
        ok = [r for r in pts if r["test_bpc"] < DIVERGED]
        if ok:
            (line,) = ax.plot([r["trainable_params"] for r in ok], [r["test_bpc"] for r in ok],
                              marker=markers[fam], label=FAMILY[fam] + (" (best lr)" if fam == "transformer" else ""))
            for r in pts:
                if r["test_bpc"] >= DIVERGED:
                    ax.scatter(r["trainable_params"], top - 0.03, marker="x", color=line.get_color(), s=50)
                    ax.annotate("diverged", (r["trainable_params"], top - 0.03), fontsize=7, xytext=(-20, -12),
                                textcoords="offset points", color=line.get_color())
        for r in results:
            if r["family"] == fam and r["variant"] not in LR_VARIANTS and r["test_bpc"] < top:
                ax.scatter(r["trainable_params"], r["test_bpc"], marker=markers[fam], facecolors="none",
                           edgecolors="gray", s=60)
                ax.annotate(r["variant"], (r["trainable_params"], r["test_bpc"]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points", color="gray")
            elif r["family"] == fam and r["variant"] not in LR_VARIANTS:
                ax.annotate(f"{r['variant']}: {r['test_bpc']:.2f} bpc", (r["trainable_params"], top - 0.01),
                            fontsize=7, xytext=(4, -10), textcoords="offset points", color="gray")
    ax.set_ylim(2.85, top)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("test bits per character (lower is better)")
    ax.set_title("Shakespeare, paper protocol" if a.protocol == "paper"
                 else f"Shakespeare, quick CPU check ({a.steps} steps x {a.batch} windows)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "bpc_vs_params.png"), dpi=130)

    if a.radii and a.protocol == "quick":
        radius_sweep(a, common)


if __name__ == "__main__":
    main()
