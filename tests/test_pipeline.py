import json
import os

import pytest
import torch

import train
from generate import generate
from utils.data_utils import encode, iterate_windows, split_tokens

TEXT = ("to be, or not to be, that is the question: whether 'tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles. ") * 80


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "tiny.txt"
    p.write_text(TEXT)
    return str(p)


def test_windows_and_splits():
    vocab = sorted(set(TEXT))
    tok = encode(TEXT, {c: i for i, c in enumerate(vocab)})
    tr, va, te = split_tokens(tok)
    assert len(tr) + len(va) + len(te) == len(tok)
    assert len(te) == len(tok) - int(len(tok) * 0.9)
    x, y = next(iterate_windows(tok, 8, 5))
    assert torch.equal(x[1, :-1], x[0, 1:]) and y[0] == tok[8]
    a = [b[0] for b in iterate_windows(tok, 8, 64, max_windows=100)]
    b = [b[0] for b in iterate_windows(tok, 8, 64, max_windows=100)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))


@pytest.mark.parametrize("args", [
    ["--model", "reservoir", "--res", "32"],
    ["--model", "reservoir", "--res", "32", "--readout", "ridge"],
    ["--model", "att_reservoir", "--res", "32", "--hid", "8"],
    ["--model", "transformer", "--hid", "32", "--heads", "2", "--layers", "1"],
])
def test_train_end_to_end(corpus, tmp_path, args):
    m, model = train.main(args + ["--data", corpus, "--seq", "16", "--emb", "8", "--epochs", "2",
                                  "--steps_per_epoch", "5", "--batch", "64", "--lr", "3e-3", "--eval_windows", "500",
                                  "--test_windows", "500", "--out_dir", str(tmp_path), "--quiet"])
    assert 0 < m["test_bpc"] < 10
    run = tmp_path / m["name"]
    assert json.load(open(run / "metrics.json"))["trainable_params"] == m["trainable_params"]
    ck = torch.load(run / "model.pt", weights_only=False)
    rebuilt = train.build_model(ck["config"], len(ck["vocab"]))
    rebuilt.load_state_dict(ck["model"])
    text = generate(rebuilt, ck["vocab"], "to be", 20, 16, generator=torch.Generator().manual_seed(0))
    assert text.startswith("to be") and len(text) == 25


def test_ridge_beats_untrained_readout(corpus, tmp_path):
    common = ["--data", corpus, "--seq", "16", "--emb", "8", "--model", "reservoir", "--res", "64",
              "--eval_windows", "1000", "--test_windows", "1000", "--out_dir", str(tmp_path), "--quiet"]
    ridge, _ = train.main(common + ["--readout", "ridge"])
    untrained, _ = train.main(common + ["--epochs", "1", "--steps_per_epoch", "1", "--lr", "1e-9"])
    assert ridge["test_bpc"] < untrained["test_bpc"] - 1.0
    assert os.path.exists(tmp_path / ridge["name"] / "model.pt")


def test_shards_partition_the_training_text():
    from utils.data_utils import shard_tokens

    tok = torch.arange(1003)
    shards = shard_tokens(tok, 5)
    assert len(shards) == 5 and torch.equal(torch.cat(shards), tok)
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_paper_protocol_preset_and_explicit_override():
    argv = ["--protocol", "paper", "--batch", "64"]
    cfg = train.apply_protocol(train.build_parser().parse_args(argv), argv)
    assert (cfg.lr, cfg.batch, cfg.shards, cfg.shard_epochs, cfg.val_frac) == (1e-4, 64, 5, 5, 0.0)
    assert cfg.test_frac == pytest.approx(1 / 6)


def test_sharded_training_without_validation(corpus, tmp_path):
    m, _ = train.main(["--data", corpus, "--model", "att_reservoir", "--res", "32", "--hid", "8", "--seq", "16",
                       "--emb", "8", "--shards", "3", "--shard_epochs", "2", "--steps_per_epoch", "3",
                       "--batch", "32", "--val_frac", "0", "--test_frac", "0.2", "--leak_min", "0.1",
                       "--warmup_steps", "5", "--test_windows", "300", "--out_dir", str(tmp_path), "--quiet"])
    h = m["history"][0]
    assert len(h["shards"]) == 6 and "val_loss" not in h and m["val_bpc"] is None
    assert 0 < m["test_bpc"] < 10
