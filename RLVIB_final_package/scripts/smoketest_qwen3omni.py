#!/usr/bin/env python
"""Step-1 smoke test for Qwen3-Omni-30B-A3B (RLVIB v0 base model).

Answers three questions before we build anything on top:
  1. Does the model load on one H200 in the `rlvib` env? (versions + GPU mem)
  2. Does it actually USE the audio track? (the answer should change when the
     audio is dropped from the *same* video)
  3. Where do the per-modality adapters live and what shapes do they output?
     -> that is exactly where the trainable fusion bottleneck will attach.

Usage (on a GPU node):
  python scripts/smoketest_qwen3omni.py --video /path/to/clip.mp4
  python scripts/smoketest_qwen3omni.py            # uses a demo URL (needs net)

Deliberately defensive: Qwen3-Omni is new, so we probe the module tree at
runtime instead of hard-coding names, and let failures surface the real API.
"""
from __future__ import annotations

import argparse
import os
import traceback

import torch

# Newer torchvision removed io.read_video, which qwen-omni-utils' default video
# backend calls. Force a working reader (override with FORCE_QWENVL_VIDEO_READER).
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_VIDEO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
QUESTION = "Describe what is happening in this clip, including any sounds you hear."


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def find_modules(model, needles):
    """Return {needle: (qualified_name, module)} for the first match per needle."""
    found = {}
    named = dict(model.named_modules())
    for needle in needles:
        for name, mod in named.items():
            if name == needle or name.endswith("." + needle):
                found[needle] = (name, mod)
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    banner("ENV")
    import transformers
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    print("transformers", transformers.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    banner("LOAD MODEL (text-only: Talker disabled)")
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model, dtype="auto", device_map="auto",
            enable_audio_output=False, attn_implementation="sdpa",
        )
    except TypeError:
        # older signature without enable_audio_output
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model, dtype="auto", device_map="auto", attn_implementation="sdpa",
        )
    if hasattr(model, "disable_talker"):
        try:
            model.disable_talker()
            print("called model.disable_talker()")
        except Exception as e:  # noqa: BLE001
            print("disable_talker() failed:", e)
    model.eval()
    if torch.cuda.is_available():
        print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    banner("MODULE TREE (top-level + thinker children)")
    print("top-level:", [n for n, _ in model.named_children()])
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        print("thinker :", [n for n, _ in thinker.named_children()])

    banner("LOCATE PER-MODALITY ADAPTERS (bottleneck attach points)")
    targets = find_modules(model, [
        "audio_tower.proj2", "audio_tower", "visual.merger", "visual",
    ])
    for needle, (name, mod) in targets.items():
        print(f"  {needle:18s} -> {name}  ({mod.__class__.__name__})")
    if not targets:
        print("  (no matches -- inspect the full tree above; API may have changed)")

    # forward hooks to capture the output shapes of the adapter modules
    shapes: dict = {}
    handles = []

    def mk_hook(tag):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) and out else out
            shapes[tag] = tuple(t.shape) if torch.is_tensor(t) else str(type(t))
        return hook

    for needle in ("audio_tower.proj2", "visual.merger"):
        if needle in targets:
            handles.append(targets[needle][1].register_forward_hook(mk_hook(needle)))

    def run(video, use_audio, tag):
        banner(f"INFERENCE [{tag}]  use_audio_in_video={use_audio}")
        from qwen_omni_utils import process_mm_info
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": QUESTION},
            ],
        }]
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=use_audio,
        )
        # move to device; cast only FLOAT tensors to the model dtype (bf16) -- leave
        # int tensors (input_ids, grids, masks) alone or embeddings/convs break.
        p = next(model.parameters())
        inputs = inputs.to(p.device)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(p.dtype)
        shapes.clear()
        with torch.no_grad():
            out = model.generate(
                **inputs, return_audio=False, thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False,
                use_audio_in_video=use_audio,
            )
        if isinstance(out, (tuple, list)):
            out = out[0]
        seq = out.sequences if hasattr(out, "sequences") else out
        gen = seq[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        print("ADAPTER OUTPUT SHAPES:", shapes)
        print("ANSWER:", answer)
        return answer

    try:
        a_av = run(args.video, True, "audio+video")
        a_v = run(args.video, False, "video-only")
        banner("AUDIO-USE CHECK")
        print("Answer changed when audio dropped? ->", a_av != a_v)
        print("(If False, the model may be ignoring audio -- the core problem we're attacking.)")
    except Exception:  # noqa: BLE001
        banner("INFERENCE FAILED -- traceback below reveals the real API")
        traceback.print_exc()
        for h in handles:
            h.remove()
        return 1

    for h in handles:
        h.remove()
    banner("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
