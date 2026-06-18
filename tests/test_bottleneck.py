"""CPU unit tests for the bottleneck modules.

Needs torch, so it is SKIPPED in the torch-free web session and runs on the cluster / in
CI with the full env. Verifies the invariants the (anchored) DPO design relies on — most
importantly that a query-conditioned VIB is still an exact identity at init (so the
`bypass`->frozen-base reference is preserved) yet genuinely conditions once trained.
"""
import pytest

torch = pytest.importorskip("torch")

from rlvib.models.bottleneck import QueryConditionedVIB, VariationalBottleneck  # noqa: E402


def _trained(b):
    """Make a freshly-built bottleneck non-trivial (out + film no longer zero)."""
    torch.nn.init.normal_(b.out.weight, std=0.1)
    torch.nn.init.normal_(b.out.bias, std=0.1)
    if hasattr(b, "film"):
        torch.nn.init.normal_(b.film.weight, std=0.1)
        torch.nn.init.normal_(b.film.bias, std=0.1)
    return b


def test_vib_identity_at_init():
    b = VariationalBottleneck(dim=16).eval()
    x = torch.randn(5, 16)
    assert torch.allclose(b(x), x, atol=1e-6)            # zero-init out -> identity


def test_qc_identity_at_init_even_with_query():
    b = QueryConditionedVIB(dim=16).eval()
    b.q = torch.randn(16)
    x = torch.randn(5, 16)
    assert torch.allclose(b(x), x, atol=1e-6)            # out AND film zero-init -> still identity


def test_qc_bypass_is_exact_identity():
    b = _trained(QueryConditionedVIB(dim=16)).eval()     # even when trained,
    b.bypass = True
    b.q = torch.randn(16)
    x = torch.randn(5, 16)
    assert torch.equal(b(x), x)                          # bypass returns x untouched (the reference)


def test_qc_query_changes_output_once_trained():
    torch.manual_seed(0)
    b = _trained(QueryConditionedVIB(dim=16)).eval()     # eval -> z=mu, deterministic
    x = torch.randn(5, 16)
    b.q = None
    y_uncond = b(x)
    b.q = torch.randn(16)
    y_q1 = b(x)
    b.q = torch.randn(16)
    y_q2 = b(x)
    assert not torch.allclose(y_uncond, y_q1)            # conditioning has an effect
    assert not torch.allclose(y_q1, y_q2)                # different query -> different edit


def test_qc_shape_and_kl_finite():
    b = QueryConditionedVIB(dim=16).train()
    b.q = torch.randn(16)
    x = torch.randn(7, 16)
    y = b(x)
    assert y.shape == x.shape
    assert torch.isfinite(b.last_kl)
    assert b.last_kl_per_token.shape == (7,)


def test_qc_batched_query_and_tokens():
    b = _trained(QueryConditionedVIB(dim=16)).eval()
    x = torch.randn(3, 7, 16)        # (B, T, dim)
    b.q = torch.randn(3, 16)         # (B, dim_q) -> FiLM broadcasts over the token axis
    y = b(x)
    assert y.shape == x.shape
