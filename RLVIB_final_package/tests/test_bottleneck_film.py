"""FiLMVariationalBottleneck: identity-at-init, conditioning plumbing, gradient flow.

The load-bearing invariant is identity-at-init: attaching the untrained prompt-aware bottleneck
must reproduce the frozen base EXACTLY (out zero-init => delta = g*out(z) = 0 => y = x), for ANY
question condition -- otherwise the anchored-DPO `bypass`->base reference is wrong. Torch is
cluster-only here, so these skip on the web session and run on the GPU node.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from rlvib.models.bottleneck import (  # noqa: E402
    FiLMVariationalBottleneck,
    VariationalBottleneck,
    set_condition,
)

DIM, HID, COND, EMBED = 16, 16, 24, 20


def _film_dict():
    bns = nn.ModuleDict({"audio": FiLMVariationalBottleneck(DIM, HID, cond_dim=COND),
                         "vision": FiLMVariationalBottleneck(DIM, HID, cond_dim=COND)})
    bns["q_proj"] = nn.Linear(EMBED, COND)
    return bns


def test_identity_at_init_unconditional():
    """No condition set => identity, for 2D (T,dim) and 3D (B,T,dim) adapter outputs."""
    bn = FiLMVariationalBottleneck(DIM, HID, cond_dim=COND).eval()
    for shape in [(7, DIM), (2, 5, DIM)]:
        x = torch.randn(*shape)
        assert torch.allclose(bn(x), x, atol=1e-5)


def test_identity_at_init_conditioned():
    """Still identity WITH a condition set (out is zero-init, regardless of q/gamma/beta/g)."""
    bns = _film_dict().eval()
    set_condition(bns, torch.randn(EMBED))
    x = torch.randn(5, DIM)
    assert torch.allclose(bns["audio"](x), x, atol=1e-5)
    # gate is computed + stashed live; sigmoid(bias=4) ~= 0.982 (open) at init
    assert bns["audio"].last_gate is not None
    assert abs(float(bns["audio"].last_gate.detach().mean()) - 0.9820) < 0.02


def test_set_condition_noop_for_plain_vib():
    """set_condition must be a safe no-op when there is no q_proj (plain/unconditional VIB)."""
    bns = nn.ModuleDict({"audio": VariationalBottleneck(DIM, HID),
                         "vision": VariationalBottleneck(DIM, HID)})
    set_condition(bns, torch.randn(EMBED))                 # no q_proj -> no-op, no error
    assert getattr(bns["audio"], "_cond", None) is None


def test_gradients_reach_new_params():
    """out moves first (delta=g*out(z), out zero-init); after one step film+gate get gradient too."""
    bns = _film_dict()
    bns.train()
    opt = torch.optim.SGD(bns.parameters(), lr=1.0)
    q = torch.randn(EMBED)
    a = bns["audio"]

    set_condition(bns, q)
    a(torch.randn(5, DIM)).sum().backward()
    assert a.out.weight.grad is not None and a.out.weight.grad.abs().sum() > 0   # out: always
    opt.step()
    opt.zero_grad()

    set_condition(bns, q)
    a(torch.randn(5, DIM)).sum().backward()
    assert a.film.weight.grad.abs().sum() > 0              # film: now that out != 0
    assert a.gate.weight.grad.abs().sum() > 0              # gate: now that out != 0


def test_eval_is_deterministic():
    """eval mode uses z=mu (no reparam noise) => repeated forwards are identical."""
    bns = _film_dict().eval()
    set_condition(bns, torch.randn(EMBED))
    x = torch.randn(5, DIM)
    assert torch.allclose(bns["audio"](x), bns["audio"](x))
