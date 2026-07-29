"""MiniCPM-O 2.6 wrapper -- a fourth audio-visual backbone (openbmb/MiniCPM-o-2_6).

Same interface contract as the Qwen wrappers (`message`, `build_inputs`, `generate`,
`adapter_modules`, `device`, `dtype`, `hidden_dim`) so the whole toolchain -- attach_bottlenecks,
train_swap_anchored, run_avhbench / run_cmm / run_mmau, the probes -- works unchanged.

MiniCPM-o-2_6 loads with trust_remote_code and its LLM is a Qwen2.5-7B (hidden 3584). The two
per-modality adapter attach points are the vision resampler and the audio projection layer.

NOTE (confirm on first cluster run, like the xattn arm): the exact module paths in
`adapter_modules()` and whether `build_inputs` must go through the model's own `.chat`
preprocessing can shift between remote-code revisions. The paths below are resolved defensively
with a fallback search; if attachment reports the wrong shapes, print `model.named_modules()` and
pin the two Linear/resampler outputs feeding the LLM.
"""
from __future__ import annotations

import torch  # noqa: E402

DEFAULT_MODEL = "openbmb/MiniCPM-o-2_6"


class MiniCPMO:
    """Thin wrapper around MiniCPM-o 2.6 for text-out audio-visual inference."""

    hidden_dim = 3584  # Qwen2.5-7B LLM width -- the adapter output dim feeding the LLM

    def __init__(self, model_id: str = DEFAULT_MODEL, attn: str = "sdpa"):
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, dtype="auto", device_map="auto",
            attn_implementation=attn,
            init_vision=True, init_audio=True, init_tts=False,      # thinker-only, no speech out
        )
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _llm(self):
        # MiniCPM-o exposes the language model under .llm (Qwen2). Fall back to the top module.
        return getattr(self.model, "llm", self.model)

    def adapter_modules(self) -> dict:
        """The per-modality adapters feeding the LLM (bottleneck attach points).

        vision -> the resampler that projects vision tokens into the LLM space;
        audio  -> the audio projection layer. Names vary by remote-code revision, so resolve
        defensively and let the caller confirm shapes on the first run.
        """
        m = self.model
        vision = (getattr(m, "resampler", None)
                  or getattr(getattr(m, "vpm", None), "resampler", None))
        audio = (getattr(m, "audio_projection_layer", None)
                 or getattr(getattr(m, "apm", None), "projector", None)
                 or getattr(m, "audio_projection", None))
        out = {}
        if vision is not None:
            out["vision"] = vision
        if audio is not None:
            out["audio"] = audio
        if not out:
            raise RuntimeError("MiniCPMO.adapter_modules: could not resolve vision/audio "
                               "projectors -- inspect model.named_modules() and pin them here.")
        return out

    @staticmethod
    def message(video=None, audio=None, prompt: str = "", fps=None) -> list:
        """Build a single user-turn conversation in MiniCPM-o's msgs format.

        Media are attached as file paths; the model's processor decodes them. No system prompt
        (benchmark convention), matching the Qwen wrappers.
        """
        content = []
        if video is not None:
            content.append({"type": "video", "video": video, **({"fps": fps} if fps else {})})
        if audio is not None:
            content.append({"type": "audio", "audio": audio})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_inputs(self, messages: list, use_audio_in_video: bool = True):
        """Tokenize + preprocess to model-ready tensors for a forward pass (grad-friendly).

        MiniCPM-o preprocessing lives in the remote code; we route through the processor the
        remote code exposes. FLOAT tensors are cast to the model dtype; int ids are left alone.
        """
        proc = getattr(self.model, "processor", None) or self.tokenizer
        inputs = proc(messages, use_audio_in_video=use_audio_in_video, return_tensors="pt")
        inputs = inputs.to(self.device)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        return inputs

    @torch.no_grad()
    def generate(self, messages: list, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        """Greedy text-out generation. Uses the model's `.chat` if present (its native path),
        else falls back to `.generate` over `build_inputs`."""
        if hasattr(self.model, "chat"):
            out = self.model.chat(msgs=messages, tokenizer=self.tokenizer,
                                  sampling=False, max_new_tokens=max_new_tokens,
                                  use_audio_in_video=use_audio_in_video)
            return (out if isinstance(out, str) else out[0]).strip()
        inputs = self.build_inputs(messages, use_audio_in_video=use_audio_in_video)
        seq = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        gen = seq[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
