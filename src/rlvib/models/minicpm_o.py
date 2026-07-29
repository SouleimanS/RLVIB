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


def _shim_transformers() -> None:
    """MiniCPM-o 2.6's remote code targets transformers ~4.44; the newer transformers this env
    needs for Qwen3-Omni removed some symbols it imports. Re-inject the ones its modeling file
    pulls in so the model loads without downgrading the whole environment. No-op if present."""
    try:
        import transformers.models.whisper.modeling_whisper as _w
        if not hasattr(_w, "WHISPER_ATTENTION_CLASSES"):
            _w.WHISPER_ATTENTION_CLASSES = {
                "eager": _w.WhisperAttention,
                "sdpa": getattr(_w, "WhisperSdpaAttention", _w.WhisperAttention),
                "flash_attention_2": getattr(_w, "WhisperFlashAttention2", _w.WhisperAttention),
            }
    except Exception:  # noqa: BLE001  -- best-effort; a real import error surfaces at load
        pass
    try:
        # newer transformers reads model.all_tied_weights_keys during load finalization;
        # MiniCPM-o's class predates it -> provide an empty default (it has no tied weights we use).
        from transformers.modeling_utils import PreTrainedModel
        if "all_tied_weights_keys" not in vars(PreTrainedModel):
            PreTrainedModel.all_tied_weights_keys = {}
    except Exception:  # noqa: BLE001
        pass
    try:
        # MiniCPM-o's forward reads cache.seen_tokens; newer transformers renamed it to
        # get_seq_length(). Re-add it as a property so the generation loop runs.
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(
                lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0)
    except Exception:  # noqa: BLE001
        pass
    try:
        # MiniCPM-o's modeling file imports flash_attn; transformers' static import check then
        # demands it installed even though we run sdpa. flash_attn needs nvcc to build (absent on
        # the login/compute nodes), so drop it from the required-imports list -- the runtime code
        # guards the flash_attn path behind is_flash_attn_2_available() and falls back to sdpa.
        import transformers.dynamic_module_utils as _dmu
        if not getattr(_dmu.get_imports, "_rlvib_noflash", False):
            _orig = _dmu.get_imports

            def _get_imports(filename):
                return [i for i in _orig(filename) if i != "flash_attn"]

            _get_imports._rlvib_noflash = True
            _dmu.get_imports = _get_imports
    except Exception:  # noqa: BLE001
        pass


class MiniCPMO:
    """Thin wrapper around MiniCPM-o 2.6 for text-out audio-visual inference."""

    hidden_dim = 3584  # Qwen2.5-7B LLM width -- the adapter output dim feeding the LLM

    def __init__(self, model_id: str = DEFAULT_MODEL, attn: str = "sdpa"):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        _shim_transformers()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # the multimodal processor (image slicing + audio features) used by the training-path
        # forward; .chat() carries its own, so generation works even if this is unavailable.
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            self.processor = None
        # No device_map="auto": MiniCPM-o's remote model class predates newer transformers'
        # accelerate device-map path (it reads all_tied_weights_keys, which the class lacks).
        # The model is ~16GB -> load it on one GPU directly, exactly as the model card does.
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            init_vision=True, init_audio=True, init_tts=False,      # thinker-only, no speech out
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

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

    def _content_list(self, messages: list, use_audio_in_video: bool = True):
        """Neutral messages -> MiniCPM-o content list: PIL frames [+ audio ndarray] + prompt."""
        video, audio, prompt = self._parse(messages)
        content = []
        if video is not None:
            content += self._encode_video(video)
            if use_audio_in_video and audio is None:
                try:
                    content.append(self._encode_audio(video))
                except Exception:  # noqa: BLE001
                    pass
        if audio is not None:
            content.append(self._encode_audio(audio))
        content.append(prompt)
        return content

    def build_inputs(self, messages: list, use_audio_in_video: bool = True):
        """Preprocess to model-ready tensors for a GRAD-CAPABLE forward (training path).

        Mirrors MiniCPM-o's own `chat()` preprocessing: media in the content list become
        `(<image>./</image>)` / `(<audio>./</audio>)` placeholders in the chat-templated prompt,
        and the images/audios go to the processor alongside. Returns the `data` dict its
        `forward(data)` consumes.
        """
        import numpy as np
        from PIL import Image

        content = self._content_list(messages, use_audio_in_video=use_audio_in_video)
        images, audios, parts = [], [], []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                parts.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                parts.append("(<audio>./</audio>)")
            else:
                parts.append(str(c))
        msgs = [{"role": "user", "content": "\n".join(parts)}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        proc = self.processor or getattr(self.model, "processor", None)
        if proc is None:
            raise RuntimeError("MiniCPMO.build_inputs: no processor available (AutoProcessor "
                               "failed to load) -- required for the training/scoring forward.")
        kw = {"return_tensors": "pt", "max_slice_nums": 1, "use_image_id": False}
        if audios:
            kw["audios"] = [audios]
        inputs = proc([prompt], [images], **kw)
        inputs = inputs.to(self.device)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        # MiniCPM-o's forward(data) reads data["position_ids"], which its TRAINING collator adds
        # (the processor does not). Supply plain 0..L-1 positions per sequence.
        if "position_ids" not in inputs and "input_ids" in inputs:
            ids = inputs["input_ids"]
            b, ln = (ids.shape[0], ids.shape[1]) if ids.dim() == 2 else (1, ids.shape[-1])
            inputs["position_ids"] = (torch.arange(ln, device=ids.device, dtype=torch.long)
                                      .unsqueeze(0).expand(b, ln))
        return inputs

    def answer_logits(self, messages: list, use_audio_in_video: bool = True):
        """Next-token logits at the answer position, WITH grad (the DPO/training hook).

        MiniCPM-o's `.chat()` is generation-only and no-grad, so scoring goes through the
        model's own `forward(data)` (which builds the multimodal embedding and runs the LLM).
        `rlvib.train.dpo.answer_logp_vec` calls this when a wrapper provides it.
        """
        inputs = self.build_inputs(messages, use_audio_in_video=use_audio_in_video)
        try:
            out = self.model(data=inputs, use_cache=False)
        except TypeError:                       # older/newer remote signature: positional dict
            out = self.model(inputs, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits[:, -1, :]

    @staticmethod
    def _parse(messages):
        """Pull (video_path, audio_path, prompt) out of the neutral message format."""
        video = audio = None
        prompt = ""
        for msg in messages:
            for c in msg.get("content", []):
                t = c.get("type")
                if t == "video":
                    video = c["video"]
                elif t == "audio":
                    audio = c["audio"]
                elif t == "text":
                    prompt = c["text"]
        return video, audio, prompt

    @staticmethod
    def _encode_video(path, fps: float = 1.0, max_frames: int = 32):
        """Sample the clip into PIL frames (MiniCPM-o's `content` wants decoded frames, not a path)."""
        import numpy as np
        from decord import VideoReader, cpu
        from PIL import Image
        vr = VideoReader(path, ctx=cpu(0))
        step = max(1, round(vr.get_avg_fps() / fps))
        idx = list(range(0, len(vr), step))
        if len(idx) > max_frames:
            idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_frames).astype(int)]
        return [Image.fromarray(f.astype("uint8")) for f in vr.get_batch(idx).asnumpy()]

    @staticmethod
    def _encode_audio(path):
        """16 kHz mono waveform (librosa reads the audio track straight from an mp4)."""
        import librosa
        y, _ = librosa.load(path, sr=16000, mono=True)
        return y

    @torch.no_grad()
    def generate(self, messages: list, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        """Greedy text-out via MiniCPM-o's native `.chat`: content = frames [+ audio] + prompt.

        Falls back to vision-only if this build's `.chat` doesn't accept an audio array in the
        content list, so eval always gets an answer (log a note rather than crash the item).
        """
        import numpy as np
        video, audio, prompt = self._parse(messages)
        content = []
        if video is not None:
            content += self._encode_video(video)
            if use_audio_in_video and audio is None:
                try:
                    content.append(self._encode_audio(video))     # audio from the video's own track
                except Exception:  # noqa: BLE001
                    pass
        if audio is not None:
            content.append(self._encode_audio(audio))
        content.append(prompt)

        def _chat(cnt):
            out = self.model.chat(msgs=[{"role": "user", "content": cnt}], tokenizer=self.tokenizer,
                                  sampling=False, max_new_tokens=max_new_tokens,
                                  use_image_id=False, max_slice_nums=1)
            return (out if isinstance(out, str) else out[0]).strip()

        import sys
        import traceback
        try:
            return _chat(content)
        except Exception:  # noqa: BLE001  -- surface the REAL stack, then retry vision-only
            print("[minicpm.generate] full content attempt failed:", file=sys.stderr)
            traceback.print_exc()
            try:
                return _chat([c for c in content if not isinstance(c, np.ndarray)])
            except Exception:  # noqa: BLE001
                print("[minicpm.generate] vision-only retry ALSO failed:", file=sys.stderr)
                traceback.print_exc()
                raise
