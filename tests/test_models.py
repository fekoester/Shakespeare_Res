import pathlib
import subprocess
import sys

import pytest
import torch

from models.att_reservoir import AttReservoirNet
from models.reservoir import ReservoirNet, spectral_radius
from models.transformer import CharTransformer
from utils.training_utils import count_parameters

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _original(module):
    """models/<module>.py (and its dependency) from the first commit, as a package `_orig_models`."""
    try:
        first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None
    pkg = pathlib.Path(__file__).parent / "_orig_models"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for m in ("reservoir", "att_reservoir"):
        code = subprocess.run(["git", "show", f"{first}:models/{m}.py"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
        (pkg / f"{m}.py").write_text(code)
    sys.path.insert(0, str(pkg.parent))
    try:
        mod = __import__(f"_orig_models.{module}", fromlist=["x"])
    finally:
        sys.path.pop(0)
    return mod


@pytest.mark.parametrize("rho", [0.5, 0.9, 1.0])
def test_spectral_radius_is_exact(rho):
    torch.manual_seed(0)
    m = ReservoirNet(30, 8, 64, spectral_radius_target=rho)
    assert spectral_radius(m.extractor.R) == pytest.approx(rho, rel=1e-5)


def test_legacy_scaling_reproduces_v01_models():
    orig_r, orig_a = _original("reservoir"), _original("att_reservoir")
    if orig_r is None:
        pytest.skip("needs git history")
    x = torch.randint(0, 30, (4, 32))
    torch.manual_seed(0)
    a = orig_r.ReservoirNet(30, 8, 64)
    torch.manual_seed(0)
    b = ReservoirNet(30, 8, 64, legacy_scaling=True)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    torch.testing.assert_close(a(x), b(x))
    torch.manual_seed(0)
    a = orig_a.AttReservoirNet(30, 8, 64, 10)
    torch.manual_seed(0)
    b = AttReservoirNet(30, 8, 64, 10, legacy_scaling=True)
    torch.testing.assert_close(a(x), b(x))
    assert spectral_radius(b.extractor.R) > 1.0  # the v0.1 bug this flag preserves


def test_only_readouts_are_trainable():
    assert count_parameters(ReservoirNet(30, 8, 64)) == 64 * 30 + 30
    m = AttReservoirNet(30, 8, 64, 10)
    assert count_parameters(m) == count_parameters(m.head)
    assert count_parameters(CharTransformer(30, 16, 2, 4, 32, 32)) > 0
