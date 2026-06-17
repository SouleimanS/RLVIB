"""Counterfactual media construction for audio-visual preference pairs (ffmpeg).

Builds the rejected / mismatch side of DPO pairs from a clean AV clip:
  Tier A (audio-drop)  : mute the audio       -> model must answer blind to audio
  Tier B (audio-swap)  : splice in another clip's audio -> seen != heard (mismatch);
                         for AVE, swap with a *different-category* clip = clean mismatch.
  Tier C (abstention)  : a Tier-B clip whose correct answer is "audio & video don't match".

See docs/research/training-data-plan.md. Requires ffmpeg on PATH.
"""
from __future__ import annotations

import os
import random
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


def make_swap_examples(items: list[dict], n: int, out_dir: str,
                       all_categories: list[str], k: int = 4,
                       rng: random.Random | None = None) -> list[dict]:
    """Materialize up to `n` audio-swapped AVE clips and build "which do you HEAR?" MCQs.

    For each base clip i (visual/seen event A) pick a clip j of a DIFFERENT category
    (audio/heard event B); write video_i + audio_j to `out_dir` (cached by name). Each
    record carries the swapped `video_path`, the seen/heard events, the MCQ, and the
    `audio_letter` (chosen) / `visual_letter` (rejected) for the contrastive DPO.
    """
    from rlvib.data import ave  # lazy: avoid any package import cycle

    rng = rng or random.Random()
    os.makedirs(out_dir, exist_ok=True)
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    order = items[:]
    rng.shuffle(order)
    out: list[dict] = []
    for it in order:
        if len(out) >= n:
            break
        other = [c for c in by_cat if c != it["category"]]
        if not other:
            continue
        jt = rng.choice(by_cat[rng.choice(other)])
        out_path = os.path.join(out_dir, f"{it['video_id']}__aud_{jt['video_id']}.mp4")
        if not os.path.exists(out_path):
            try:
                swap_audio(it["video_path"], jt["video_path"], out_path)
            except subprocess.CalledProcessError:
                continue
        mcq = ave.make_hear_mcq(jt["category"], it["category"], all_categories, k=k, rng=rng)
        out.append(dict(mcq, video_path=out_path))
    return out
