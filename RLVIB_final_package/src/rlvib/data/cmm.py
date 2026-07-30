"""CMM loader — "The Curse of Multi-Modalities" (arXiv:2410.12787, DAMO-NLP-SG).

HF dataset `DAMO-NLP-SG/CMM`. Flat JSON of yes/no hallucination probes; each clip
contributes a "yes" probe (object/event present) and a "no" probe (absent ->
hallucination test). Fields: category, sub_category, modality ("visual"/"audio"),
granularity, video_path, audio_path (relative, e.g. "./raw_files/.../id.mp4"),
question (already ends "Answer with yes or no."), answer ("yes"/"no").

6 sub_categories x 200 samples (400 Q) each. Audio-touching subsets:
"""
from __future__ import annotations

import json
import os

from torch.utils.data import Dataset

AUDIO_SUBSETS = ("audio-language", "visual-audio-language", "overrely_audio_ignore_visual")


class CMMDataset(Dataset):
    """Yields {video_path, audio_path, question, answer, sub_category, modality}."""

    def __init__(self, json_path: str, data_root: str, sub_categories=None):
        with open(json_path) as f:
            samples = json.load(f)
        if sub_categories:
            keep = set(sub_categories)
            samples = [s for s in samples if s.get("sub_category") in keep]
        self.samples = samples
        self.data_root = data_root

    def _resolve(self, p):
        if not p:
            return None
        return os.path.join(self.data_root, p.lstrip("./"))  # "./raw_files/.." -> root/raw_files/..

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "video_path": self._resolve(s.get("video_path")),
            "audio_path": self._resolve(s.get("audio_path")),
            "question": s["question"],
            "answer": str(s["answer"]).strip().lower(),  # "yes" / "no"
            "sub_category": s.get("sub_category"),
            "modality": s.get("modality"),
        }
