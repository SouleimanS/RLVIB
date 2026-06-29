"""AVE (Audio-Visual Event) loader — clean video+audio substrate for v0 training.

YapengTian/AVE-ECCV18. ~4,143 ten-second clips over 28 categories; each clip is
annotated with the [start, end] seconds where the event is BOTH audible and visible.
  videos       : <root>/AVE/{video_id}.mp4   (named by YouTube id)
  annotations  : <root>/Annotations.txt      (Category&VideoID&Quality&Start&End)
  splits       : <root>/{trainSet,valSet,testSet}.txt   (same line format)

Audio is salient + visually grounded by construction -> a clean substrate for
audio-dependent QA + audio-swap counterfactual pairs (see training-data-plan.md).
"""
from __future__ import annotations

import os
import random
import string

DEFAULT_ROOT = "data/AVE/AVE_Dataset"
_SPLIT_FILE = {"train": "trainSet.txt", "val": "valSet.txt",
               "test": "testSet.txt", "all": "Annotations.txt"}


def _parse_line(line: str):
    p = line.rstrip("\n").split("&")
    if len(p) < 5 or p[0] == "Category":
        return None
    try:
        return {"category": p[0], "video_id": p[1], "quality": p[2],
                "start_s": int(p[3]), "end_s": int(p[4])}
    except ValueError:
        return None


def load_ave(split: str = "train", root: str = DEFAULT_ROOT) -> list[dict]:
    """Return [{video_path, category, video_id, start_s, end_s}] for a split
    (train/val/test/all), skipping clips whose mp4 isn't on disk."""
    vdir = os.path.join(root, "AVE")
    items = []
    with open(os.path.join(root, _SPLIT_FILE[split])) as f:
        for line in f:
            rec = _parse_line(line)
            if rec is None:
                continue
            vp = os.path.join(vdir, f"{rec['video_id']}.mp4")
            if os.path.exists(vp):
                rec["video_path"] = vp
                items.append(rec)
    return items


def categories(root: str = DEFAULT_ROOT) -> list[str]:
    """Sorted set of the 28 event categories."""
    cats = set()
    with open(os.path.join(root, "Annotations.txt")) as f:
        for line in f:
            rec = _parse_line(line)
            if rec:
                cats.add(rec["category"])
    return sorted(cats)


def make_mcq(category: str, all_categories: list[str], k: int = 4,
             rng: random.Random | None = None) -> dict:
    """Audio-visual MCQ: which event is BOTH seen and heard? (audio-dependent).

    Returns {question, options, answer, gold_index, gold_letter}.
    """
    rng = rng or random.Random()
    distractors = rng.sample([c for c in all_categories if c != category],
                             min(k - 1, len(all_categories) - 1))
    options = distractors + [category]
    rng.shuffle(options)
    gi = options.index(category)
    return {
        "question": "Which event is BOTH visible and audible in this clip?",
        "options": options,
        "answer": category,
        "gold_index": gi,
        "gold_letter": string.ascii_uppercase[gi],
    }


def make_hear_mcq(audio_event: str, visual_event: str, all_categories: list[str],
                  k: int = 4, rng: random.Random | None = None) -> dict:
    """MCQ for an audio-SWAPPED clip: "which event do you HEAR?".

    Options always include BOTH the heard event (gold) and the seen event (the visual
    shortcut), plus distractors. Returns the usual fields plus `audio_letter` (gold)
    and `visual_letter` (the seen shortcut) so the DPO can contrast them.
    """
    rng = rng or random.Random()
    pool = [c for c in all_categories if c not in (audio_event, visual_event)]
    distractors = rng.sample(pool, min(k - 2, len(pool)))
    options = [audio_event, visual_event] + distractors
    rng.shuffle(options)
    ai, vi = options.index(audio_event), options.index(visual_event)
    return {
        "question": "Which event do you HEAR in this clip?",
        "options": options,
        "audio_event": audio_event,
        "visual_event": visual_event,
        "audio_letter": string.ascii_uppercase[ai],
        "visual_letter": string.ascii_uppercase[vi],
        "gold_letter": string.ascii_uppercase[ai],
    }


def make_see_mcq(audio_event: str, visual_event: str, all_categories: list[str],
                 k: int = 4, rng: random.Random | None = None) -> dict:
    """Vision-question counterpart of make_hear_mcq for an audio-SWAPPED clip: "which do you SEE?".

    Same option set (both events + distractors), but the SEEN event is gold. For the DPO this
    flips the roles vs the HEAR question on the SAME clip: chosen = `visual_letter` (seen = the
    correct answer to a SEE question), rejected = `audio_letter` (the heard event = the audio
    distractor the gate must SUPPRESS when vision is the trusted stream). Pairing a HEAR and a
    SEE example on one swapped clip is the contrast that forces a question-conditioned gate g(q)
    to route by question type instead of re-collapsing to the unconditional vision rewrite.
    """
    rng = rng or random.Random()
    pool = [c for c in all_categories if c not in (audio_event, visual_event)]
    distractors = rng.sample(pool, min(k - 2, len(pool)))
    options = [audio_event, visual_event] + distractors
    rng.shuffle(options)
    ai, vi = options.index(audio_event), options.index(visual_event)
    return {
        "question": "Which event do you SEE in this clip?",
        "options": options,
        "audio_event": audio_event,
        "visual_event": visual_event,
        "audio_letter": string.ascii_uppercase[ai],
        "visual_letter": string.ascii_uppercase[vi],
        "gold_letter": string.ascii_uppercase[vi],   # seen = gold (vs heard = gold for HEAR)
    }


def format_mcq(question: str, options: list[str]) -> str:
    """Lettered prompt (parseable by rlvib.eval.metrics.parse_choice)."""
    opts = "\n".join(f"({string.ascii_uppercase[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}\nAnswer with only the letter."
