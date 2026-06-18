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
        # VideoLLaMA2 loads fp16 for inference, but is TRAINED in bf16 (every finetune script:
        # --bf16 True --fp16 False). A differentiable fp16 forward overflows attention
        # (Q.K^T > 65504 -> NaN), and an fp16 + bf16-autocast hybrid mis-scales the frozen
        # projector. So run the whole stack in bf16 (the LLaVA recipe). Also force-load the
        # delay_load SigLIP tower (model_init already does; this is belt-and-suspenders).
        try:
            vt = self.model.get_model().get_vision_tower()
            if not getattr(vt, "is_loaded", True):
                vt.load_model()
        except Exception:  # noqa: BLE001 -- best-effort; model_init force-loads vision anyway
            pass
        self.model = self.model.to(torch.bfloat16)
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

    # System prompt for the chat template; mirrors mm_infer's eval-time prompt. Verify
    # against the installed videollama2 package if the answers look off.
    _SYS = ("You are a helpful language and vision assistant. You are able to understand "
            "the visual content that the user provides, and assist the user with a variety "
            "of tasks using natural language.")

    def build_inputs(self, message: dict, use_audio_in_video: bool = True) -> dict:
        """Teacher-forced inputs for a forward that returns answer-position logits.

        Mirrors mm_infer's preprocessing but targets model.forward (keeps grad): the
        bottleneck hooks on the mm projectors fire during multimodal embedding, and
        logits[:, -1, :] is the next-token (answer) distribution. Runs in the rlvib_vl2
        env; UNTESTED on GPU -- validate and iterate.
        """
        from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN
        from videollama2.mm_utils import tokenizer_multimodal_token

        video, audio = message.get("video"), message.get("audio")
        prompt = message.get("prompt", "")
        if video is not None:
            media = self.processor["video"](video, va=use_audio_in_video)
            modal, modal_token = "video", DEFAULT_VIDEO_TOKEN
        elif audio is not None:
            media = self.processor["audio"](audio)
            modal, modal_token = "audio", DEFAULT_AUDIO_TOKEN
        else:
            raise ValueError("VideoLLaMA2 build_inputs needs a video or audio input")

        if isinstance(media, dict):                       # AV model: {"video","audio"} feats
            media = {k: v.to(self.device, self.dtype) for k, v in media.items()}
        else:
            media = media.to(self.device, self.dtype)
        images = [(media, modal)]

        conv = [{"role": "system", "content": self._SYS},
                {"role": "user", "content": modal_token + "\n" + prompt}]
        text = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(text, self.tokenizer, modal_token,
                                               return_tensors="pt").unsqueeze(0).to(self.device)
        return {"input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "images": images}

    @torch.no_grad()
    def generate(self, message: dict, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        # Reuse build_inputs (bf16 media) so eval matches training; mm_infer hardcodes
        # .half() (fp16), which would mismatch the bf16 model.
        inputs = self.build_inputs(message, use_audio_in_video)
        out = self.model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            images=inputs["images"], do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[0].strip()
