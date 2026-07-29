"""DAVE loader (arXiv:2503.09321, gorjanradevski/dave).

The HF dataset ships the clips inside ego4d.zip / epic.zip (no EPIC/Ego4D access
needed). Its loader is a dataset *script* (dave.py), unsupported by datasets>=4, so
we read ego4d.json / epic.json directly and resolve media against the extracted zip.

Per `dave.py` _info, each record has media fields + `choice_metadata[mode] =
{choices: [str], ground_truth: int}`. We evaluate one `mode` at a time; the four
single-choice modes share the question and differ only in the media -> that gives
the modality-ablation ΔAcc:
  audio_visual_alignment -> video_with_overlayed_audio_path  (video, audio on)
  visual_only            -> silent_video_path                (video, audio off)
  audio_only             -> overlayed_audio_path             (audio only)
  text_only              -> (no media)                       (language prior)
"""
from __future__ import annotations

import json
import os

from torch.utils.data import Dataset

MODE_SPEC = {
    "audio_visual_alignment": {"field": "video_with_overlayed_audio_path", "kind": "video", "use_audio": True},
    "visual_only": {"field": "silent_video_path", "kind": "video", "use_audio": False},
    "audio_only": {"field": "overlayed_audio_path", "kind": "audio", "use_audio": False},
    "text_only": {"field": None, "kind": None, "use_audio": False},
}


class DaveDataset(Dataset):
    """Yields {media_path, kind, use_audio, choices, gt_index, audio_class, type}."""

    def __init__(self, json_path: str, media_root: str, mode: str = "audio_visual_alignment"):
        if mode not in MODE_SPEC:
            raise ValueError(f"mode must be one of {list(MODE_SPEC)}")
        with open(json_path) as f:
            self.samples = json.load(f)
        self.media_root = media_root
        self.mode = mode
        self.spec = MODE_SPEC[mode]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cm = s["choice_metadata"][self.mode]
        field = self.spec["field"]
        media = os.path.join(self.media_root, s[field]) if field and s.get(field) else None
        return {
            "media_path": media,
            "kind": self.spec["kind"],          # "video" | "audio" | None
            "use_audio": self.spec["use_audio"],
            "choices": cm["choices"],
            "gt_index": cm["ground_truth"],     # int index into choices
            "audio_class": s.get("audio_class"),
            "type": s.get("type"),
        }
