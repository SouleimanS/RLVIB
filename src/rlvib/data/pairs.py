"""Counterfactual media construction for audio-visual preference pairs (ffmpeg).

Builds the rejected / mismatch side of DPO pairs from a clean AV clip:
  Tier A (audio-drop)  : mute the audio       -> model must answer blind to audio
  Tier B (audio-swap)  : splice in another clip's audio -> seen != heard (mismatch);
                         for AVE, swap with a *different-category* clip = clean mismatch.
  Tier C (abstention)  : a Tier-B clip whose correct answer is "audio & video don't match".

See docs/research/training-data-plan.md. Requires ffmpeg on PATH.
"""
from __future__ import annotations

import subprocess


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def mute_audio(video_in: str, out_path: str) -> str:
    """Tier A: drop the audio track (silent video)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_in,
          "-an", "-c:v", "copy", out_path])
    return out_path


def swap_audio(video_in: str, audio_src: str, out_path: str) -> str:
    """Tier B/C: replace `video_in`'s audio with the audio of `audio_src`.

    `audio_src` may be a video (its audio track is used) or an audio file.
    """
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_in, "-i", audio_src,
          "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
          "-shortest", out_path])
    return out_path
