"""CrossModalAttention: identity-at-init, bypass, mask routing, gradient flow.

Same load-bearing contract as every other arm: attaching the untrained module must
reproduce the frozen base exactly (zero-init out projections), and bypass must be a
no-op. Torch is cluster-only; these skip on the web session.
"""
import pytest

torch = pytest.importorskip("torch")

from rlvib.models.xattn import CrossModalAttention  # noqa: E402

D, S = 32, 20


def _mod_and_masks():
    mod = CrossModalAttention(D, heads=4)
    a_mask = torch.zeros(S, dtype=torch.bool)
    v_mask = torch.zeros(S, dtype=torch.bool)
    a_mask[3:8] = True                                  # 5 audio positions
    v_mask[10:16] = True                                # 6 vision positions
    return mod, a_mask, v_mask


def test_identity_at_init():
    mod, a_mask, v_mask = _mod_and_masks()
    mod.eval()
    mod.set_masks(a_mask, v_mask)
    x = torch.randn(1, S, D)
    assert torch.allclose(mod(x), x, atol=1e-6)          # zero-init out projections


def test_bypass_and_no_masks():
    mod, a_mask, v_mask = _mod_and_masks()
    x = torch.randn(1, S, D)
    assert mod(x) is x                                   # no masks set -> untouched
    mod.set_masks(a_mask, v_mask)
    mod.bypass = True
    assert mod(x) is x                                   # bypass -> untouched
    mod.bypass = False
    bad = torch.zeros(S + 5, dtype=torch.bool)           # seq-length mismatch (decode step)
    mod.set_masks(bad, bad)
    assert mod(x) is x


def test_edit_touches_only_media_positions_and_grads_flow():
    mod, a_mask, v_mask = _mod_and_masks()
    mod.train()
    # push the out projections off zero so the edit is non-trivial
    for mha in (mod.a_from_v, mod.v_from_a):
        torch.nn.init.normal_(mha.out_proj.weight, std=0.05)
    mod.set_masks(a_mask, v_mask)
    x = torch.randn(1, S, D)
    y = mod(x)
    text_mask = ~(a_mask | v_mask)
    assert torch.allclose(y[0, text_mask], x[0, text_mask])          # text untouched
    assert not torch.allclose(y[0, a_mask], x[0, a_mask])            # audio edited
    assert not torch.allclose(y[0, v_mask], x[0, v_mask])            # vision edited
    y.sum().backward()
    assert mod.a_from_v.in_proj_weight.grad.abs().sum() > 0          # grads reach the attention
    assert mod.v_from_a.in_proj_weight.grad.abs().sum() > 0
