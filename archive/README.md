# archive/ — retired experiments (kept, not deleted)

Code that is **not** part of the story in `paper/phd_talk.tex`. Nothing on the active path
imports these; they are preserved for provenance and can be restored with `git mv` if revived.

| folder | what it was | why retired |
|---|---|---|
| `localization/` | cosine/saliency **localization maps** (`aligner.py`, `train_aligner`, `localize_cosine`, `extract_maps`, `visual_attention`, `vib_saliency`) + its research note | a separate line of work (where in the frame the model looks); the deck uses the attention-share and likelihood-shift probes instead |
| `dave/` | the **DAVE** benchmark (`data/dave.py`, `run_dave.py`, `eval_dave.qsub`) | not one of the deck's benchmarks (AVHBench / CMM / MMAU / Video-MME / sync) |
| `debug/` | one-off **VideoLLaMA2 NaN** debug scripts | their finding is written up in `docs/research/videollama2-fp16-bf16.md`; the fix lives in the VIB (`normalize_input`) |

The core `train/dpo.py` still contains `grpo_step` (GRPO shares the fixed-eps replay path with the
VIB), and `eval/contrastive.py` is imported by both eval runners — so those stay in place even
though GRPO/contrastive decoding are not headline results.
