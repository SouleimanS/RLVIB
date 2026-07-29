"""Temporal clock features + sync-record construction.

Pure-python sync-record checks always run; the TemporalVariationalBottleneck identity/bypass
contract skips without torch (cluster-only), matching the other model tests.
"""
import pytest

from rlvib.data.pairs import _sync_mcq


def test_sync_record_fields():
    shifted = _sync_mcq(True, 1.0)                       # audio lags (delta>0)
    assert shifted["chosen_letter"] == "B" and shifted["rejected_letter"] == "A"
    assert shifted["dir_letter"] == "A" and shifted["shifted"] is True
    lead = _sync_mcq(True, -2.0)                         # audio leads (delta<0)
    assert lead["dir_letter"] == "B"
    synced = _sync_mcq(False, 0.0)
    assert synced["chosen_letter"] == "A" and "dir_letter" not in synced


torch = pytest.importorskip("torch")

from rlvib.models.temporal import TemporalVariationalBottleneck, clock_features  # noqa: E402

D, S = 32, 20


def test_clock_features_shape_and_range():
    clk = clock_features(S, period=50.0)
    assert clk.shape == (S, 2)
    assert float(clk.abs().max()) <= 1.0 + 1e-6


def test_temporal_identity_at_init():
    mod = TemporalVariationalBottleneck(D).eval()       # W_out zero-init -> identity
    x = torch.randn(1, S, D)
    assert torch.allclose(mod(x), x, atol=1e-6)


def test_temporal_bypass_and_output_width():
    mod = TemporalVariationalBottleneck(D)
    x = torch.randn(1, S, D)
    mod.bypass = True
    assert mod(x) is x                                   # bypass -> exact base
    mod.bypass = False
    assert mod(x).shape == x.shape                       # output stays at token width `dim`
