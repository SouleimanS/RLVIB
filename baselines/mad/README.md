# MAD baseline (Modality-Adaptive Decoding, Chung et al., KAIST, arXiv:2601.21181)

**Training-free** decoding for cross-modal hallucination. Per item:

1. **Modality-weight extraction** — append *"To answer this question, which modality is
   needed (audio, video, or both)?"*, read the model's next-token logits for
   `both`/`video`/`audio`, softmax → (w_av, w_v, w_a).
2. **Modality-adaptive generation** (paper Eq. 9) — combine four contrastive branches per
   decoded token from the four input configurations (clean / video-removed / audio-removed /
   both-removed) with strengths γ·w, γ = 2.5 (paper Sec. 4.1.4); greedy argmax.

Perturbation here = **modality removal** (one of the paper's stated corruption choices):
`v′` drops the video (keeps the clip's audio track as an audio input), `a′` drops the audio
(`use_audio_in_video=False`), `v′a′` is text-only. Answers are scored with the repo's usual
suffix and yes/no parser, so numbers are directly comparable with our tables.

**Cost:** ~5 forwards per decoded token (4 configs + the weight pass). Use `--limit`.

## Run

```bash
PYTHONPATH=src python baselines/mad/eval_mad.py --bench avhbench --model qwen2.5-omni --limit 300
PYTHONPATH=src python baselines/mad/eval_mad.py --bench cmm      --model qwen3-omni   --limit 300
# -> runs/mad_<bench>_<model>.json (resumable; per-task + overall accuracy inside)
```

MAD is decoding-side, so it also **composes** with our trained adapters: attach a checkpoint
first (load_attached) and MAD's four branches all run through the adapted model — a
`MAD + FiLM` row is one extra experiment.
