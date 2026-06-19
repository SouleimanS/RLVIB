"""Closed-API frontier baselines (Gemini, GPT-4o) for benchmarking ONLY.

These are reference baselines, NOT trainable: closed APIs expose no weights, no forward
hooks, and no gradients, so the per-modality VIB cannot be attached (adapter_modules /
device / dtype raise). They implement just `message` + `generate`, conform to the eval
harness, and run CPU-only with internet (e.g. an ABCI login node) -- never on the
air-gapped GPU batch nodes. Keys come from env: GEMINI_API_KEY / OPENAI_API_KEY.

  Gemini  : native audio-visual -- uploads the clip (video carries its audio track) via
            the Files API; the honest AV frontier reference.
  GPT-4o  : NO unified non-speech audio-visual input. Default = sampled video frames
            (vision only); set OPENAI_AUDIO=1 with an audio-capable model id to also send
            raw audio. Vision-only GPT-4o will (correctly) fail audio-presence probes.

Deps (login node, rlvib env): pip install google-genai openai ; ffmpeg on PATH.
UNTESTED on this machine (no keys/network here) -- validate with scripts/api_smoketest.py
before a full run. See also docs and the eval modules rlvib.eval.run_{cmm,avhbench,dave}.
"""
from __future__ import annotations

import base64
import os
import subprocess
import time


def _retry(fn, tries=4, base=2.0, what="api call"):
    """Call fn(); on exception retry with exponential backoff (rate limits / transient)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- providers raise many error types; backoff on all
            if i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"  [{what}] {type(e).__name__}: {e} -- retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    return None


class _APIModel:
    """Shared message() + a hard stop for any attempt to attach a bottleneck."""

    hidden_dim = None

    @staticmethod
    def message(video=None, audio=None, prompt: str = "") -> dict:
        return {"video": video, "audio": audio, "prompt": prompt}

    def adapter_modules(self):
        raise NotImplementedError(
            "Closed API models are frozen black boxes: no weights/hooks/gradients, so the "
            "VIB cannot be attached. Use them for base benchmarking only (no --bottleneck).")

    @property
    def device(self):
        raise NotImplementedError("API model: no local device.")

    @property
    def dtype(self):
        raise NotImplementedError("API model: no local dtype.")


# --------------------------------------------------------------------------- Gemini

class GeminiModel(_APIModel):
    """Google Gemini via the google-genai SDK (native video+audio understanding)."""

    def __init__(self, model_id: str | None = None):
        from google import genai  # google-genai package

        self.model_id = model_id or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (or GOOGLE_API_KEY) for the Gemini baseline")
        self.client = genai.Client(api_key=key)

    def _upload(self, path: str):
        """Upload a media file and block until it is ACTIVE (video needs processing)."""
        f = self.client.files.upload(file=path)
        for _ in range(60):
            state = getattr(getattr(f, "state", None), "name", str(getattr(f, "state", "")))
            if state == "ACTIVE":
                return f
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing FAILED for {path}")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        return f

    def generate(self, message: dict, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        from google.genai import types

        uploaded, parts = [], []
        for path in (message.get("video"), message.get("audio")):
            if path and os.path.exists(path):
                f = _retry(lambda p=path: self._upload(p), what="gemini upload")
                uploaded.append(f)
                parts.append(f)
        parts.append(message.get("prompt", ""))
        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max(max_new_tokens, 16))
        try:
            resp = _retry(lambda: self.client.models.generate_content(
                model=self.model_id, contents=parts, config=cfg), what="gemini generate")
            return (getattr(resp, "text", None) or "").strip()
        finally:
            for f in uploaded:  # keep the Files quota clean (20GB / 48h auto-expiry)
                try:
                    self.client.files.delete(name=f.name)
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- GPT-4o

def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _sample_frames_b64(path: str, n: int) -> list[str]:
    """n evenly-spaced JPEG frames (base64) via ffmpeg piping -- no temp files."""
    dur = _ffprobe_duration(path)
    times = [dur * (i + 0.5) / n for i in range(n)] if dur > 0 else [0.0]
    frames = []
    for t in times:
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path,
             "-frames:v", "1", "-vf", "scale=512:-2", "-f", "image2", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True)
        if out.returncode == 0 and out.stdout:
            frames.append(base64.b64encode(out.stdout).decode())
    return frames


def _audio_wav_b64(path: str) -> str | None:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        capture_output=True)
    return base64.b64encode(out.stdout).decode() if out.returncode == 0 and out.stdout else None


class OpenAIModel(_APIModel):
    """OpenAI GPT-4o via chat.completions. Vision = sampled frames; audio is opt-in.

    GPT-4o has no single-call audio-visual input for non-speech audio, so the default is
    vision-only (sampled frames). Set OPENAI_AUDIO=1 AND an audio-capable OPENAI_MODEL
    (e.g. gpt-4o-audio-preview) to also attach raw audio; otherwise audio probes are
    expected to fail (an honest limitation, not a bug).
    """

    def __init__(self, model_id: str | None = None):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY for the GPT-4o baseline")
        self.model_id = model_id or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI()
        self.n_frames = int(os.environ.get("OPENAI_FRAMES", "8"))
        self.with_audio = os.environ.get("OPENAI_AUDIO", "0") == "1"

    def generate(self, message: dict, use_audio_in_video: bool = True,
                 max_new_tokens: int = 256) -> str:
        content = [{"type": "text", "text": message.get("prompt", "")}]
        video = message.get("video")
        if video and os.path.exists(video):
            for b64 in _sample_frames_b64(video, self.n_frames):
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        if self.with_audio and use_audio_in_video:
            src = message.get("audio") or video
            wav = _audio_wav_b64(src) if (src and os.path.exists(src)) else None
            if wav:
                content.append({"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}})

        def _call():
            return self.client.chat.completions.create(
                model=self.model_id, temperature=0.0, max_tokens=max(max_new_tokens, 16),
                messages=[{"role": "user", "content": content}])

        resp = _retry(_call, what="openai generate")
        return (resp.choices[0].message.content or "").strip()
