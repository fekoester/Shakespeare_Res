"""Character-level Transformer sweep on Shakespeare (v0.1 configurations). Extra arguments are
passed to train.py for every configuration."""

import sys

import train

TRANSFORMER_CONFIGS = [(64, 4, 4), (72, 8, 8), (128, 8, 8), (356, 8, 8), (256, 16, 16)]  # (ffn size, heads, layers)

if __name__ == "__main__":
    for hid, heads, layers in TRANSFORMER_CONFIGS:
        train.main(["--model", "transformer", "--hid", str(hid), "--heads", str(heads), "--layers", str(layers),
                    "--epochs", "5", "--lr", "1e-4", *sys.argv[1:]])
