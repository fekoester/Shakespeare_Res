"""Classic reservoir on Shakespeare (v0.1 configuration). Extra arguments are passed to train.py,
e.g. `python train_reservoir.py --readout ridge` or `--legacy_scaling`."""

import sys

import train

if __name__ == "__main__":
    train.main(["--model", "reservoir", "--res", "750", "--epochs", "5", "--lr", "1e-4", *sys.argv[1:]])
