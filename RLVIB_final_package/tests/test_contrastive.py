"""CPU unit test for the audio-aware contrastive-decoding logit combine.

`contrastive_answer` needs a model + GPU, but `contrastive_logits` is pure tensor math, so we
pin its behaviour here: the (1+alpha)*full - alpha*no_audio combine and the VCD plausibility
mask that -inf's out tokens the full pass already finds implausible (so the contrast can't
promote a degenerate token). torch is cluster-only, so skip cleanly where it isn't installed.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from rlvib.eval.contrastive import contrastive_logits  # noqa: E402  (after importorskip)


def test_combine_no_plausibility():
    lf = torch.tensor([[1.0, 2.0, 3.0]])
    ln = torch.tensor([[0.0, 2.0, 1.0]])
    cd = contrastive_logits(lf, ln, alpha=1.0, plausibility=0.0)
    assert torch.allclose(cd, torch.tensor([[2.0, 2.0, 5.0]]))   # (1+1)*full - 1*no_audio


def test_alpha_zero_is_identity():
    lf = torch.tensor([[1.0, 5.0, -2.0]])
    ln = torch.tensor([[3.0, 0.0, 4.0]])
    cd = contrastive_logits(lf, ln, alpha=0.0, plausibility=0.0)
    assert torch.allclose(cd, lf)                                # no_audio term vanishes


def test_plausibility_masks_implausible_tokens():
    # full strongly prefers index 2; the no-audio pass would otherwise drag index 0 way up.
    lf = torch.tensor([[-10.0, 0.0, 5.0]])
    ln = torch.tensor([[100.0, 0.0, 0.0]])
    cd = contrastive_logits(lf, ln, alpha=1.0, plausibility=0.1)
    # floor = max(full) + log(0.1) = 5 - 2.30 = 2.70; only index 2 (=5.0) clears it.
    assert math.isinf(cd[0, 0].item()) and cd[0, 0].item() < 0
    assert math.isinf(cd[0, 1].item()) and cd[0, 1].item() < 0
    assert torch.isfinite(cd[0, 2])
    assert int(cd.argmax(dim=-1)[0]) == 2                        # the plausible, audio-favoured token


def test_promotes_audio_favoured_token():
    # both tokens plausible under full; token 1 rises with audio, token 0 falls.
    lf = torch.tensor([[2.0, 2.0]])
    ln = torch.tensor([[3.0, 1.0]])                              # no-audio prefers token 0
    cd = contrastive_logits(lf, ln, alpha=1.0, plausibility=0.5)
    assert int(cd.argmax(dim=-1)[0]) == 1
