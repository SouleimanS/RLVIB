"""Qwen3-Omni thinker-only wrapper — the v0 base model for RLVIB.

Productizes the verified smoke-test recipe (see scripts/smoketest_qwen3omni.py and
docs/research/grounding-audio-to-video.md "Step 1 results"):
  - load thinker-only (enable_audio_output=False + disable_talker)
  - build inputs via qwen_omni_utils.process_mm_info, cast FLOAT tensors to bf16
  - greedy text-only generation

The per-modality adapters whose outputs the fusion bottleneck will wrap are
exposed via `adapter_modules()`:
  audio  -> thinker.audio_tower.proj2  (Linear)            -> (T_a, 2048)
  vision -> thinker.visual.merger      (VisionPatchMerger) -> (T_v, 2048)
"""
from __future__ import annotations

import os

# qwen-omni-utils' default video backend calls torchvision.io.read_video, which
# newer torchvision removed; force a working reader before it is imported.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import torch  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class QwenOmni:
    """Thin wrapper around Qwen3-Omni for text-out audio-visual inference."""

    hidden_dim = 2048  # adapter output dim (audio_tower.proj2 / visual.merger)

    def __init__(self, model_id: str = DEFAULT_MODEL, attn: str = "sdpa"):
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        self.model_id = model_id
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        try:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_id, dtype="auto", device_map="auto",
                enable_audio_output=False, attn_implementation=attn,
            )
        except TypeError:  # older signature without enable_audio_output
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_id, dtype="auto", device_map="auto", attn_implementation=attn,
            )
        if hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
            except Exception:  # noqa: BLE001
                pass
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def adapter_modules(self) -> dict:
        """The per-modality adapters feeding the Thinker (bottleneck attach points)."""
        t = self.model.thinker
        return {"audio": t.audio_tower.proj2, "vision": t.visual.merger}

    @staticmethod
    def message(video=None, audio=None, prompt: str = "") -> list:
        """Build a single user-turn conversation for the processor."""
        content = []
        if video is not None:
            content.append({"type": "video", "video": video})
        if audio is not None:
            content.append({"type": "audio", "audio": audio})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_inputs(self, messages: list, use_audio_in_video: bool = True):
        from qwen_omni_utils import process_mm_info

        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.device)
        for k, v in list(inputs.items()):  # cast FLOAT tensors only; leave int ids/grids
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        from rlvib.models.bottleneck import condition_bottlenecks
        condition_bottlenecks(self, messages)  # no-op unless a QueryConditionedVIB is attached
        return inputs

    @torch.no_grad()
    def generate(self, messages: list, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        inputs = self.build_inputs(messages, use_audio_in_video=use_audio_in_video)
        out = self.model.generate(
            **inputs, return_audio=False, thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=max_new_tokens, thinker_do_sample=False,
            use_audio_in_video=use_audio_in_video,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        seq = out.sequences if hasattr(out, "sequences") else out
        gen = seq[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
