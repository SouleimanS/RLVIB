"""Qwen2.5-Omni-7B thinker-only wrapper — RLVIB comparison arm.

Shares the rlvib env with Qwen3-Omni (Qwen2.5-Omni landed in transformers 4.52 and
remains in 5.x). Uses the Thinker-only class so no Talker is loaded. Adapters:
`audio_tower.proj` (1280->3584) and `visual.merger` (->3584); hidden dim 3584
(vs 2048 for Qwen3-Omni). API surface (process_mm_info, messages) matches Qwen3-Omni.
"""
from __future__ import annotations

import os

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import torch  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"


class Qwen25Omni:
    """Thin wrapper around Qwen2.5-Omni (Thinker-only) for text-out AV inference."""

    hidden_dim = 3584  # adapter output dim (audio_tower.proj / visual.merger)

    def __init__(self, model_id: str = DEFAULT_MODEL, attn: str = "sdpa"):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.model_id = model_id
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id, dtype="auto", device_map="auto", attn_implementation=attn,
        )
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def adapter_modules(self) -> dict:
        return {"audio": self.model.audio_tower.proj, "vision": self.model.visual.merger}

    @staticmethod
    def message(video=None, audio=None, prompt: str = "") -> list:
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
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        return inputs

    @torch.no_grad()
    def generate(self, messages: list, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        inputs = self.build_inputs(messages, use_audio_in_video=use_audio_in_video)
        try:
            out = self.model.generate(
                **inputs, thinker_max_new_tokens=max_new_tokens, do_sample=False,
                use_audio_in_video=use_audio_in_video,
            )
        except TypeError:  # fall back to the standard kwarg
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_audio_in_video=use_audio_in_video,
            )
        if isinstance(out, (tuple, list)):
            out = out[0]
        seq = out.sequences if hasattr(out, "sequences") else out
        gen = seq[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
