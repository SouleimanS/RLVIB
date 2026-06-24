"""The Qwen-Omni wrappers must inject the official system prompt (AVHBench/CMM parity).

The wrappers `import torch` at module load (cluster-only), so skip cleanly where it isn't
installed; `message()` itself is pure dict-building and needs no model weights.
"""
import pytest

pytest.importorskip("torch")


def _check(message_fn, sysprompt):
    msg = message_fn(video="clip.mp4", prompt="Is there a dog barking?")
    assert msg[0]["role"] == "system"
    assert msg[0]["content"][0]["text"] == sysprompt
    assert "virtual human" in sysprompt                      # the official Qwen line, verbatim
    assert msg[1]["role"] == "user"
    types = [c["type"] for c in msg[1]["content"]]           # user turn still carries the inputs
    assert "video" in types and "text" in types
    # fps, when given, rides along on the video content (matches the standalone harness)
    vid = next(c for c in message_fn(video="clip.mp4", prompt="x", fps=1.0)[1]["content"]
               if c["type"] == "video")
    assert vid["fps"] == 1.0


def test_qwen25_omni_system_prompt():
    from rlvib.models.qwen25_omni import SYSTEM_PROMPT, Qwen25Omni
    _check(Qwen25Omni.message, SYSTEM_PROMPT)


def test_qwen3_omni_has_no_system_prompt():
    # Qwen3-Omni sets NO system prompt for benchmark eval (repo README "Evaluation"); user-only turn.
    from rlvib.models.qwen3_omni import QwenOmni
    msg = QwenOmni.message(video="clip.mp4", prompt="Is there a dog barking?", fps=1.0)
    assert all(m["role"] != "system" for m in msg)
    assert msg[0]["role"] == "user"
    types = [c["type"] for c in msg[0]["content"]]
    assert "video" in types and "text" in types
    vid = next(c for c in msg[0]["content"] if c["type"] == "video")   # fps still rides along
    assert vid["fps"] == 1.0
