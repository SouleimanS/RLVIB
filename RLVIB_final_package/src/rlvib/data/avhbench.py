"""AVHBench loader (arXiv:2410.18325, ICLR 2025).

Distributed via Google Drive (not on the HF Hub). The QA file is a flat JSON list
of {video_id, task, text, label}; videos are {video_id}.mp4 with audio embedded,
in a flat directory. Tasks (exact `task` strings, verified from the released json):
  "Video-driven Audio Hallucination"  -> asks about audio          (label Yes/No)
  "Audio-driven Video Hallucination"  -> asks about video          (label Yes/No)
  "AV Matching"                       -> do audio & video match?   (label Yes/No)
  "AV Captioning"                     -> free-text caption
"""
from __future__ import annotations

import json
import os

from torch.utils.data import Dataset

BINARY_TASKS = (
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Matching",
)
CAPTION_TASK = "AV Captioning"


class AVHBenchDataset(Dataset):
    """Yields {video_path, task, text, label} per QA item."""

    def __init__(self, qa_json: str, video_root: str, tasks=None):
        with open(qa_json) as f:
            samples = json.load(f)
        if tasks is not None:
            keep = set(tasks)
            samples = [s for s in samples if s.get("task") in keep]
        self.samples = samples
        self.video_root = video_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "video_path": os.path.join(self.video_root, f"{s['video_id']}.mp4"),
            "task": s["task"],
            "text": s["text"],
            "label": s["label"],
        }
