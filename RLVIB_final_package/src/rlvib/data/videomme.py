"""Video-MME loader (Fu et al., CVPR 2025, arXiv:2405.21075).

900 YouTube videos x 3 QA = 2,700 four-way MCQs over three duration buckets --
short (<2 min), medium (4-15 min), long (30-60 min). We evaluate the standard
**without-subtitles** track; the audio track is fed to the model straight from the
video file (use_audio_in_video), which is exactly the capability our adapters target.
This is the capability guard at video-understanding scale: a trained adapter must not
cost general video QA accuracy (the "alignment tax" check).

Layout (download on a LOGIN node -- see scripts/eval_videomme.qsub header):
  data/VideoMME/videomme/test-00000-of-00001.parquet     # annotations (HF: lmms-lab/Video-MME)
  data/VideoMME/videos/<videoID>.mp4                     # unzipped videos (videoID = YouTube id)
"""
from __future__ import annotations

import glob
import os

DEFAULT_DIR = "data/VideoMME"
DURATIONS = ("short", "medium", "long")

_PROMPT = ("Select the best answer to the following multiple-choice question based on the video. "
           "Respond with only the letter (A, B, C, or D) of the correct option.")


def _read_annotations(data_dir: str) -> list[dict]:
    """The HF dump is parquet; read via pandas (datasets pulls it in), else `datasets`."""
    pats = (os.path.join(data_dir, "videomme", "*.parquet"), os.path.join(data_dir, "*.parquet"))
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"no Video-MME parquet under {data_dir} (see module docstring)")
    try:
        import pandas as pd
        return [r for f in files for r in pd.read_parquet(f).to_dict("records")]
    except ImportError:
        from datasets import load_dataset
        return list(load_dataset("parquet", data_files=list(files))["train"])


def _resolve(video_id: str, root: str) -> str:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = os.path.join(root, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(root, f"{video_id}.mp4")    # let the eval surface a clean "missing file"


def load_videomme(data_dir: str = DEFAULT_DIR, video_root: str | None = None,
                  durations: list[str] | None = None) -> list[dict]:
    """[{video_path, question, options, answer, duration, task_type, domain, id}] in file order.

    `durations`: subset of {"short","medium","long"} (None = all). Keep the runner's
    resume logic in mind: one output file per durations setting, since it indexes by order.
    """
    video_root = video_root or os.path.join(data_dir, "videos")
    keep = {d.strip().lower() for d in durations} if durations else None
    items = []
    for s in _read_annotations(data_dir):
        dur = str(s.get("duration", "")).lower()
        if keep and dur not in keep:
            continue
        vid = s.get("videoID") or s.get("video_id")
        items.append({
            "video_path": _resolve(str(vid), video_root),
            "question": s["question"],
            "options": [str(o) for o in (s.get("options") if s.get("options") is not None else [])],
            "answer": str(s.get("answer", "")).strip().upper(),          # gold letter "A".."D"
            "duration": dur or "unknown",
            "task_type": str(s.get("task_type", "")),
            "domain": str(s.get("domain", "")),
            "id": str(s.get("question_id") or vid),
        })
    return items


def format_videomme(question: str, options: list[str]) -> str:
    """Official Video-MME answer prompt (options arrive pre-lettered "A. ...").

    Parse the reply with rlvib.eval.metrics.parse_choice.
    """
    return _PROMPT + "\n" + question + "\n" + "\n".join(options)
