"""LoRA baseline: identity-at-init, bypass = exact base, gradient flow, hook composition.

The load-bearing invariant mirrors the VIB's: attaching the untrained LoRA must reproduce
the frozen base EXACTLY (B zero-init => delta = 0), and `bypass` must return the base output
so the anchored-DPO reference works unchanged. Torch is cluster-only; skips on the web session.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from rlvib.models.lora import LoRAAdapter, attach_lora  # noqa: E402


class _StubModel:
    """Minimal wrapper exposing the attach contract (adapter_modules/model/device/dtype)."""

    def __init__(self):
        self.model = nn.Sequential()                       # params to freeze (none needed)
        self.audio = nn.Linear(8, 8)                       # adapter IS a Linear (qwen3 audio case)
        self.vision = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def adapter_modules(self):
        return {"audio": self.audio, "vision": self.vision}


def test_identity_at_init_and_bypass():
    m = _StubModel()
    x = torch.randn(4, 8)
    base_a, base_v = m.audio(x), m.vision(x)
    ad, handles = attach_lora(m, r=4, alpha=8)
    assert set(ad.keys()) == {"audio", "vision_0", "vision_2"}   # every Linear wrapped
    assert torch.allclose(m.audio(x), base_a, atol=1e-6)         # B zero-init => identity
    assert torch.allclose(m.vision(x), base_v, atol=1e-6)
    for a in ad.values():                                        # bypass => still exact base
        a.bypass = True
    assert torch.allclose(m.audio(x), base_a, atol=1e-6)
    for h in handles:
        h.remove()


def test_gradient_flow_and_effect():
    m = _StubModel()
    ad, handles = attach_lora(m, r=4, alpha=8)
    opt = torch.optim.SGD(ad.parameters(), lr=1.0)
    x = torch.randn(4, 8)
    m.audio(x).sum().backward()
    assert ad["audio"].B.weight.grad.abs().sum() > 0             # grad reaches the LoRA delta
    opt.step()
    base = nn.Linear(8, 8)
    base.load_state_dict(m.audio.state_dict())
    assert not torch.allclose(m.audio(x), base(x))               # after a step, output moved
    ad["audio"].bypass = True
    assert torch.allclose(m.audio(x), base(x), atol=1e-6)        # bypass recovers exact base
    for h in handles:
        h.remove()


def test_total_kl_and_state_dict_roundtrip():
    from rlvib.models.bottleneck import set_bypass, total_kl
    m = _StubModel()
    ad, handles = attach_lora(m, r=4, alpha=8)
    assert total_kl(ad) == 0.0                                   # no IB rate term for LoRA
    set_bypass(ad, True)                                         # trainer helper works as-is
    assert all(a.bypass for a in ad.values())
    m2 = _StubModel()
    ad2, h2 = attach_lora(m2, r=4, alpha=8)
    ad2.load_state_dict(ad.state_dict())                         # checkpoint roundtrip
    for h in handles + h2:
        h.remove()


def test_adapter_shapes():
    a = LoRAAdapter(8, 16, r=4, alpha=8)
    assert a(torch.randn(3, 8)).shape == (3, 16)
    assert a.scale == 2.0
