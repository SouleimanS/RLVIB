"""CPU smoke test for the GRPO bottleneck step (docs/research/grpo-vib.md).

`grpo_step` needs torch but not a GPU or any model weights: it only calls answer_logp_vec /
set_bypass / total_kl. We stand up a tiny stub LM that applies the *real* VariationalBottleneck
modules to a fixed pseudo-hidden (mimicking the per-modality adapter forward hooks), so the RL
math -- group sampling, ternary reward, group-relative advantage, the two KL terms, and the
optimizer step -- runs end-to-end on CPU. Skips cleanly where torch isn't installed (web parity).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rlvib.models.bottleneck import (  # noqa: E402
    VariationalBottleneck,
    capture_eps,
    set_forced_eps,
)
from rlvib.train.dpo import grpo_step  # noqa: E402

DIM, VOCAB = 16, 32
YES, NO, ABSTAIN = 1, 2, 3


class _TinyLM(nn.Module):
    """Frozen 'base': a fixed hidden -> (per-modality bottlenecks) -> linear head -> logits."""

    def __init__(self, bottlenecks):
        super().__init__()
        self.bns = bottlenecks
        self.head = nn.Linear(DIM, VOCAB)
        self.register_buffer("h0", torch.randn(1, 1, DIM))

    def forward(self, **_inputs):
        h = self.h0
        for b in self.bns.values():          # mimic the adapter forward hooks (audio, then vision)
            h = b(h)
        return SimpleNamespace(logits=self.head(h))   # (1, 1, VOCAB)


class _StubModel:
    """Minimal surface answer_logp_vec touches: .model (the LM), .dtype, .build_inputs."""

    def __init__(self, bottlenecks):
        self.model = _TinyLM(bottlenecks)
        self.dtype = torch.float32

    def build_inputs(self, _messages, use_audio_in_video=True):
        return {}                            # the stub LM ignores inputs


def _setup():
    torch.manual_seed(0)
    bns = nn.ModuleDict({"audio": VariationalBottleneck(DIM), "vision": VariationalBottleneck(DIM)})
    model = _StubModel(bns)
    opt = torch.optim.AdamW(bns.parameters(), lr=1e-2)
    return model, bns, opt


def _batch(gold="yes", abstain=True):
    ex = {"messages": [{"role": "user", "content": "is there a dog barking?"}],
          "gold": gold, "yes_id": YES, "no_id": NO}
    if abstain:
        ex["abstain_id"] = ABSTAIN
    return [ex, dict(ex, gold="no")]


def test_grpo_step_runs_and_updates():
    model, bns, opt = _setup()
    before = [p.detach().clone() for p in bns.parameters()]

    m = grpo_step(model, bns, opt, _batch(), group=8)

    # metrics present and finite
    assert set(m) == {"reward", "adv_std", "kl_vib", "kl_ref", "p_correct"}
    for k, v in m.items():
        assert v == v and abs(v) < 1e6, (k, v)           # not NaN/inf
    assert -1.0 <= m["reward"] <= 1.0
    assert 0.0 <= m["p_correct"] <= 1.0
    assert m["kl_vib"] >= -1e-4                           # a KL rate is non-negative
    assert m["adv_std"] >= 0.0

    # the optimizer step actually moved the bottleneck (policy-gradient and/or KL terms)
    after = [p.detach() for p in bns.parameters()]
    moved = sum(float((a - b).abs().sum()) for a, b in zip(after, before))
    assert moved > 0.0


def test_grpo_step_two_way_no_abstain():
    """Omitting abstain_id collapses to plain yes/no; reward is then +/-1 only."""
    model, bns, opt = _setup()
    m = grpo_step(model, bns, opt, _batch(abstain=False), group=6)
    assert m["reward"] == m["reward"]                     # finite
    assert -1.0 <= m["reward"] <= 1.0


def test_grpo_step_drgrpo_advantage():
    """Dr. GRPO path (std_norm=False, mean-centered advantage) runs and moves the bottleneck."""
    model, bns, opt = _setup()
    before = [p.detach().clone() for p in bns.parameters()]
    m = grpo_step(model, bns, opt, _batch(), group=8, std_norm=False)
    for v in m.values():
        assert v == v and abs(v) < 1e6
    moved = sum(float((a.detach() - b).abs().sum()) for a, b in zip(bns.parameters(), before))
    assert moved > 0.0


def test_grpo_step_restores_eval_mode():
    """A step taken from eval() must leave the bottlenecks back in eval() (sampling is internal)."""
    model, bns, opt = _setup()
    bns.eval()
    grpo_step(model, bns, opt, _batch(), group=4)
    assert not bns.training


def test_forced_eps_replays_deterministically():
    """Pinning eps must reproduce the exact z/logits (the fixed-eps replay grpo_step relies on)."""
    torch.manual_seed(0)
    bns = nn.ModuleDict({"audio": VariationalBottleneck(DIM), "vision": VariationalBottleneck(DIM)})
    for b in bns.values():
        nn.init.normal_(b.out.weight, std=0.1)   # un-zero the residual so eps reaches the output
    bns.train()
    x = torch.randn(1, 1, DIM)

    y1 = torch.stack([b(x) for b in bns.values()])      # free-running sample
    eps = capture_eps(bns)
    assert set(eps) == {"audio", "vision"}

    set_forced_eps(bns, eps)                             # replay -> identical output
    y2 = torch.stack([b(x) for b in bns.values()])
    assert torch.allclose(y1, y2, atol=1e-6)

    set_forced_eps(bns, None)                            # cleared -> resamples (differs)
    y3 = torch.stack([b(x) for b in bns.values()])
    assert not torch.allclose(y1, y3, atol=1e-6)
