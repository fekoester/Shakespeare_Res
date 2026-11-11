"""Attention-enhanced reservoir sweep on Shakespeare (v0.1 configurations). Extra arguments are
passed to train.py for every configuration."""

import sys

import train

ATT_CONFIGS = [(75, 13), (75, 19), (100, 20), (150, 25), (160, 30)]  # (reservoir size, readout size)

if __name__ == "__main__":
    for res, hid in ATT_CONFIGS:
        train.main(["--model", "att_reservoir", "--res", str(res), "--hid", str(hid), "--epochs", "5",
                    "--lr", "1e-4", *sys.argv[1:]])
