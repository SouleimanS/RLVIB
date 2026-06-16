"""VideoLLaMA2.1-7B-AV wrapper — RLVIB comparison arm.

Runs ONLY in the rlvib_vl2 env (transformers==4.42.3; see environment-vllama2.yml),
which is incompatible with the Qwen-Omni env. Different API (model_init / mm_infer).
UNTESTED until the vl2 env + repo + checkpoint are set up. Adapters:
`model.model.mm_projector` (video / STC connector) + `model.model.mm_projector_a`
(audio MLP); hidden dim 3584. Audio is embedded in the mp4 (extracted with va=True).
"""
from __future__ import annotations

import torch

DEFAULT_MODEL = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"


class VideoLLaMA2:
    """Wrapper around VideoLLaMA2.1-7B-AV (model_init / mm_infer)."""

    hidden_dim = 3584

    def __init__(self, model_id: str = DEFAULT_MODEL):
        from videollama2 import model_init
        from videollama2.utils import disable_torch_init

        disable_torch_init()
        self.model, self.processor, self.tokenizer = model_init(model_id)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def adapter_modules(self) -> dict:
        m = self.model.model  # Videollama2Qwen2Model
        return {"audio": m.mm_projector_a, "vision": m.mm_projector}

    @staticmethod
    def message(video=None, audio=None, prompt: str = "") -> dict:
        return {"video": video, "audio": audio, "prompt": prompt}

    @torch.no_grad()
    def generate(self, message: dict, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        from videollama2 import mm_infer

        video, audio = message.get("video"), message.get("audio")
        prompt = message.get("prompt", "")
        if video is not None:
            tensor = self.processor["video"](video, va=use_audio_in_video)
            modal = "video"
        elif audio is not None:
            tensor = self.processor["audio"](audio)
            modal = "audio"
        else:
            raise ValueError("VideoLLaMA2 needs a video or audio input")
        return mm_infer(tensor, prompt, model=self.model, tokenizer=self.tokenizer,
                        modal=modal, do_sample=False).strip()
