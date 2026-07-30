# Getting the bottleneck to train on VideoLLaMA2 (the fp16 saga) — diagnosis & fix

Deep-research synthesis (5 angles, adversarially verified). Resolves why our differentiable
training forward on frozen VideoLLaMA2.1-7B-AV blew up (NaN logits, then loss=inf) while the same
model evals fine. Sourcing note: arXiv/blogs 403'd the fetch tool; primary code claims were read
from the live VideoLLaMA2 / LLaVA repos via `raw.githubusercontent.com` (high confidence).

## TL;DR

Two independent bugs, both from **running an fp16-loaded model through a training forward**:

1. **NaN logits = fp16 attention overflow.** `Q·Kᵀ` exceeds fp16's 65 504 ceiling → `softmax(inf)=NaN`.
   VideoLLaMA2 is **trained in bf16** (every finetune script: `--bf16 True --fp16 False`); fp16 is only
   the *inference* loader. My "fp16 weights + bf16 autocast" patch is the worst config (PyTorch
   maintainer ptrblck: autocast assumes fp32 master params and only casts op inputs, never widening
   fp16 weights). **Fix: run the whole stack in bf16** (the LLaVA recipe `model.to(bfloat16)`).
2. **loss=inf = the VIB's `mu²` term exploding** on **massive activations** (~1e9 features; Sun et al.
   2024). Only **input normalization** fixes it: `logvar` clamp caps only `exp(logvar)`, KL free-bits is
   a *floor* not a ceiling, grad-clip can't rescale an already-exploded grad, and lowering β doesn't kill
   a 1e19. Because `W_out` is zero-init, the blow-up hits on the **first backward from the KL term,
   before `W_out` matters** — so the fix must be on the encoder *input*.

## What each angle established

- **Forward vs generate (angle 1):** `forward` and `generate` call the *identical*
  `prepare_inputs_labels_for_multimodal` — so a manual `forward(images=[(media,modal)])` is supported and
  does **not** skip the vision tower. The frozen **STCConnectorV35 projector has no internal dtype cast**,
  and the vision tower casts its output back to the *media tensor's* dtype; `mm_infer` rigorously feeds
  **fp16 media** (`.half()`), keeping the projector fp16 → sane O(1) features. Wrapping the forward in
  `autocast(bf16)` breaks that discipline → mis-scaled output.
  (`videollama2_qwen2.py`, `videollama2_arch.py`, `encoder.py`, `projector.py`, audio_visual branch.)
- **Vision tower warning (angle 2):** the SigLIP tower is `delay_load=True`, but **`model_init` *does*
  force-load it** (`"videollama" in config.model_type` is True), and the *"weights not used when
  initializing"* warning is **benign** (collaborator-confirmed, VideoLLaMA2 issue #118; LLaVA #672). So
  the tower is loaded in our wrapper — not the cause. (We still force-load it explicitly as insurance.)
- **Mixed precision (angle 3):** bf16 has fp32's dynamic range (no overflow, **no GradScaler**); fp16
  needs loss scaling and overflows >65 504. fp16-weights+bf16-autocast is a non-standard hybrid. The ~1e9
  features are likely **genuine activations fp16 was silently clipping** (1e9 ≫ 65 504, ≪ bf16's 3.4e38)
  — bf16 is the messenger, not the cause. (PyTorch AMP docs; HF perf docs; ptrblck.)
- **Official recipe (angle 4):** every VideoLLaMA2 AV/audio finetune script is
  `--bf16 True --tf32 True --fp16 False`; model + both towers loaded/cast to **bf16**; the training forward
  is the **labels-based `model(input_ids, attention_mask, labels, images)`** with `images` = a list of
  `(tensor, modal)` tuples **pre-moved to bf16**. Issue #165 shows the exact `loss→nan→0.0` overflow
  signature. (VideoLLaMA2 `train.py:453/487/498/552/596`; `scripts/custom/*.sh`.)
- **Scale-robust VIB (angle 5):** adapters (Houlsby/Pfeiffer) **LayerNorm the encoder input** while the
  residual keeps raw `x` — scale-invariant `mu`, unchanged residual semantics. **Vittle** (NeurIPS 2025)
  uses our *exact* KL with no input-norm, but on **RMSNorm'd LLM hidden states** (normal scale) — not raw
  projector outputs; that's precisely our Qwen-vs-VideoLLaMA2 split. Input-LN **changes the trained
  function**, so gate it **per-backbone** (off for Qwen → its results stand; on for VideoLLaMA2).

## The fix as implemented

- **`models/videollama2.py`:** force-load the SigLIP tower, then `self.model.to(torch.bfloat16)`;
  `build_inputs` feeds **bf16** media; `generate` reimplemented to reuse `build_inputs` (bf16) instead of
  `mm_infer` (which hardcodes `.half()`).
- **`models/bottleneck.py`:** `VariationalBottleneck(normalize_input=…)` — parameter-free `F.layer_norm`
  on the **encoder input only** (residual keeps raw `x`); threaded through `attach_bottlenecks` /
  `load_attached` and saved in the checkpoint. **Off for Qwen (byte-for-byte unchanged), on for
  VideoLLaMA2.**
- **`train/dpo.py`:** `clip_grad_norm_(max_norm=1.0)` before `optimizer.step()` — inert at Qwen's scale,
  spike insurance for VideoLLaMA2.
- **`train_swap_anchored.py`:** `normalize_input` auto-on for `videollama2`; saved in the checkpoint.

## Sources

- VideoLLaMA2 (audio_visual): `videollama2_qwen2.py`, `videollama2_arch.py`, `encoder.py`, `projector.py`,
  `train.py`, `scripts/custom/*.sh`, issues [#118](https://github.com/DAMO-NLP-SG/VideoLLaMA2/issues/118),
  [#165](https://github.com/DAMO-NLP-SG/VideoLLaMA2/issues/165) ·
  [LLaVA train.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py), issue
  [#672](https://github.com/haotian-liu/LLaVA/issues/672)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) · [PyTorch mixed-precision blog](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) · [ptrblck: explicit cast vs autocast](https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618)
- [Massive Activations in LLMs (Sun et al. 2024)](https://arxiv.org/abs/2402.17762) · [Vittle / Visual Instruction Bottleneck Tuning (NeurIPS 2025)](https://arxiv.org/abs/2505.13946) · [Deep VIB (Alemi et al. 2017)](https://arxiv.org/abs/1612.00410) · [Houlsby adapters](https://arxiv.org/abs/1902.00751)
