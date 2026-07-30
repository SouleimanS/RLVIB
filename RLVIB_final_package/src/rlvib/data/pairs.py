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


def silence_audio(video_in: str, out_path: str) -> str:
    """Replace the audio with digital silence (keeps an audio track so the encoder runs)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_in,
          "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
          "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
          "-shortest", out_path])
    return out_path


def swap_video(video_in: str, video_src: str, out_path: str) -> str:
    """Swap-video counterpart of `swap_audio`: keep `video_in`'s AUDIO, take the VIDEO
    stream from `video_src` -> heard != seen. Re-encodes video (resolutions may differ)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_in, "-i", video_src,
          "-map", "1:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
          "-c:a", "aac", "-shortest", out_path])
    return out_path


def shift_audio(video_in: str, delta: float, out_path: str) -> str:
    """Displace the audio track by `delta` seconds relative to the video (temporal intervention).

    delta > 0 -> audio LAGS (arrives late); delta < 0 -> audio LEADS (arrives early). Implemented
    by re-muxing the same file's audio with an `-itsoffset`; the video stream is copied.
    """
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_in,
          "-itsoffset", f"{delta}", "-i", video_in,
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
          "-shortest", out_path])
    return out_path


def make_video_swap_examples(items: list[dict], n: int, out_dir: str,
                             all_categories: list[str], k: int = 4,
                             rng: random.Random | None = None) -> list[dict]:
    """Symmetric to `make_swap_examples`, but swaps the VIDEO: keep clip i's audio (heard A),
    splice in a different-category clip j's video (seen B) -> "which do you SEE?" MCQ with
    `visual_letter` (chosen = the seen event) vs `audio_letter` (rejected = the heard event).

    Targets the A->V axis (video grounding). See the deck's symmetric-experiment slides.
    """
    from rlvib.data import ave

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
        out_path = os.path.join(out_dir, f"{it['video_id']}__vid_{jt['video_id']}.mp4")
        if not os.path.exists(out_path):
            try:
                swap_video(it["video_path"], jt["video_path"], out_path)
            except subprocess.CalledProcessError:
                continue
        # audio (heard) = it.category ; video (seen, swapped-in) = jt.category
        mcq = ave.make_see_mcq(it["category"], jt["category"], all_categories, k=k, rng=rng)
        out.append(dict(mcq, video_path=out_path,
                        chosen_letter=mcq["visual_letter"], rejected_letter=mcq["audio_letter"]))
    return out


# 2-way / 3-way sync-judgment MCQ text (no ave dependency -- the axes are fixed)
_SYNC_OPTIONS = ("audio and video are in sync", "audio and video are out of sync")
_DIR_OPTIONS = ("the audio lags (comes late)", "the audio leads (comes early)")


def _sync_mcq(shifted: bool, delta: float) -> dict:
    """Build a balanced sync-judgment record. chosen = the correct axis answer; rejected =
    the visual prior ("in sync"). For shifted clips a direction (lags/leads) is also recorded."""
    opts = list(_SYNC_OPTIONS)
    chosen = "B" if shifted else "A"                     # A=in-sync, B=out-of-sync
    rec = {"question": "Are the audio and the video synchronized?", "options": opts,
           "sync_letter": chosen, "prior_letter": "A", "shifted": shifted, "delta": delta,
           "chosen_letter": chosen, "rejected_letter": "A"}
    if shifted:
        rec["dir_options"] = list(_DIR_OPTIONS)
        rec["dir_letter"] = "A" if delta > 0 else "B"    # A=lags (delta>0), B=leads (delta<0)
    return rec


def make_shift_examples(items: list[dict], n: int, out_dir: str,
                        offsets=(0.5, 1.0, 2.0), rng: random.Random | None = None) -> list[dict]:
    """Materialize up to `n` temporally-shifted clips and build sync-judgment records for
    shift-DPO. Half the records are the untouched clip (gold "in sync"), half are shifted by a
    random +/- offset (gold "out of sync", with a lags/leads direction). Balanced => chance 0.50.

    chosen = correct sync answer, rejected = the "in sync" visual prior -> preferring the correct
    answer requires reading the timing, not the visual prior.
    """
    rng = rng or random.Random()
    os.makedirs(out_dir, exist_ok=True)
    order = items[:]
    rng.shuffle(order)
    out: list[dict] = []
    for it in order:
        if len(out) >= n:
            break
        make_shifted = len(out) % 2 == 1                 # alternate synced / shifted -> balanced
        if not make_shifted:
            out.append(dict(_sync_mcq(False, 0.0), video_path=it["video_path"]))
            continue
        delta = rng.choice(offsets) * rng.choice((-1.0, 1.0))
        tag = f"{delta:+.1f}".replace(".", "p")
        out_path = os.path.join(out_dir, f"{it['video_id']}__shift_{tag}.mp4")
        if not os.path.exists(out_path):
            try:
                shift_audio(it["video_path"], delta, out_path)
            except subprocess.CalledProcessError:
                continue
        out.append(dict(_sync_mcq(True, delta), video_path=out_path))
    return out


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
        out.append(dict(mcq, video_path=out_path,
                        chosen_letter=mcq["audio_letter"], rejected_letter=mcq["visual_letter"]))
    return out
