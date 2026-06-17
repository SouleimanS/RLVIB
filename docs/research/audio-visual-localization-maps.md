# Faithful audio→video localization maps on a frozen AV-LLM + bottleneck

Deep-research synthesis (5 angles: SSL feature-similarity · AV cross-attention/segmentation ·
gradient/Grad-CAM · MLLM attention attribution · metrics & controls). Sourcing caveat: the
research agents hit HTTP 403 on direct PDF fetches, so *mechanism* claims read from authors'
GitHub code are high-confidence; exact benchmark decimals are medium-confidence.

## Bottom line

1. The "where is the sound" heatmap that the SSL literature produces is, mechanically, a
   **cosine-similarity map**: L2-normalize a pooled audio vector and each visual patch vector,
   dot-product per patch, upsample. Senocak's official code does exactly this with **both
   backbones frozen** and only a small projection trained — i.e. our "frozen base + small
   trainable head" design *is* the canonical recipe.
2. Our paradigm (freeze the big model, train a tiny head that reads attention between a query
   token and the (h,w) patch tokens) is published and works: **F-LMM (CVPR 2025)**. And
   grounding lives in **a few "localization heads"** you *select*, not average — which is why
   blanket per-token KL was mush.
3. For an attention readout, **raw/averaged attention is NOT faithful** (fails the ICML-2022
   Faithfulness-Violation Test; decoder-only models dump attention on BOS/"sink" tokens).
   **Attention×gradient (Chefer)** — one backward pass, training-free — is the faithful readout.
4. **The trap, confirmed by ~5 independent papers: visual-saliency bias.** Audio-visual models
   mostly localize the salient object and ignore the audio; vision-only baselines *match or beat*
   audio-visual ones, and ~90% of benchmark clips are solvable with no sound. **So a map that
   looks right on a matched clip proves nothing.** The result only counts if it passes
   audio-dependence controls — and our **audio-swap clips are exactly that control.**
5. Practical data fact: **AVE has only temporal labels (when), no spatial boxes (where).** For
   quantitative spatial validation we need **VGG-SS / AVSBench / IS3+** (IS3+ ships
   silence/noise/offscreen negatives ready-made).

---

## 1. What actually produces a faithful map (ranked)

### A. Cosine-similarity map (canonical, frozen-compatible, ~free) — START HERE
- **Mechanism (code-verified across the lineage):** `map[patch] = cos(â, v_patch)` where `â` is a
  pooled audio embedding and `v_patch` each spatial visual vector; ReLU/softmax; bilinear-upsample.
  - Senocak CVPR'18: L2-norm both → matmul → ReLU → softmax over 20×20 grid; **VGG16 + SoundNet
    both frozen**, only two FC layers trained. (`github.com/ardasnck/learning_to_localize_sound_source`)
  - LVS CVPR'21: `A = einsum('ncqa,nchw->nqa', img_norm, aud_norm)`. (`arxiv.org/abs/2104.02691`)
  - EZ-VSL ECCV'22: `einsum('nchw,mc->nmhw')`, MIL-max alignment (audio matches ≥1 patch).
    (`arxiv.org/abs/2203.09324`)
- **Frozen-backbone SSL is SOTA:** recent methods build the localizer on **frozen CLIP ViT + BEATs**,
  training only an alignment module. (`arxiv.org/abs/2504.15118`)
- **Caveat:** requires audio & visual tokens to live in a *comparable* space. Qwen's audio/visual
  adapter outputs both feed the same LLM (both ~2048-d) so they may be roughly comparable — but a
  small **trainable projection (our bottleneck)** is the principled way to align them.

### B. Attention×gradient on the frozen LLM (faithful attention readout, training-free)
- **Don't trust raw attention:** "Attention is not Explanation" (`aclanthology.org/N19-1357`); the
  Faithfulness-Violation Test finds raw attention & rollout fail polarity-consistency (~40% on
  multimodal models), while **Attention⊙Gradient is best almost everywhere**
  (`proceedings.mlr.press/v162/liu22i/liu22i.pdf`).
- **Recipe (Chefer):** `Ā = E_h[(∇A ⊙ A)⁺]` rolled across layers; backprop an audio-grounded target
  to attention, positive-clamp, mean over a *selected* head subset, take the visual-token columns →
  patch grid. (`arxiv.org/abs/2103.15679`, `github.com/hila-chefer/Transformer-MM-Explainability`)
- **Localization heads:** ~3 of thousands of heads rival fine-tuned grounding; select by low
  spatial-entropy and **filter BOS/sink + edge-row heads.** (`arxiv.org/abs/2503.06287`)
- **F-LMM precedent:** freeze the LMM, train only a small U-Net head over the attention stack →
  competitive RefCOCO segmentation. (`arxiv.org/abs/2406.05821`)

### C. Trainable cross-attention bottleneck (our contribution, needs training + faithfulness work)
- **AVSegFormer** = audio-as-query cross-attention over visual tokens → strong AVS localizer
  (`arxiv.org/abs/2307.01146`); **TPAVI** = audio-modulated non-local block on the visual grid
  (`arxiv.org/abs/2207.05042`). Both fine-tune backbones (not frozen) — so they validate the
  *mechanism*, not our frozen constraint.
- The attention matrix `[audio, patch]` *is* the localization map by construction — but only if
  trained so the audio query genuinely gates it (see controls). The fix that makes audio matter is
  **contrastive**: separate high- vs low-audio-response regions (`arxiv.org/abs/2503.12847`, CVPR'25).

### D. Gradient/Grad-CAM (training-free, weaker, noisier) — cross-check only
- Plain Grad-CAM passes Adebayo sanity checks but is coarse/noisy; **Guided variants FAIL** sanity
  checks (model-independent edge detectors — do not use). (`papers.neurips.cc/paper/8160`)
- Audio-specific version = **counterfactual difference**: attribute (audio-present − silence) via
  Integrated-Gradients-with-counterfactual-baseline. (`arxiv.org/abs/2109.13412`)
- In dedicated SSL, the field abandoned CAM for similarity maps (they localize better).

---

## 2. The recipe that fits our setup (and snaps into existing code)

The pieces we already have line up almost perfectly:
- **The bottleneck → an audio-visual *alignment* head.** Instead of a per-modality VIB that
  compresses tokens (gave diffuse KL), make it a small projection (or cross-attention) that maps
  audio tokens and visual patch tokens into a comparable space; the **cosine / attention matrix is
  the map.**
- **The swap pairs → both the contrastive training signal AND the audit** (Section 3).

Steps:
1. On a **true matched** clip, grab frozen adapter tokens: audio `A`, visual `V` reshaped to (t,h,w).
2. **Free baseline (no training):** `M = cos(mean(A), V)` per patch → upsample → overlay (renderer
   ready). Also compute the attention×gradient map. *This immediately tells us if Qwen's tokens are
   comparable enough to localize at all.*
3. **Trainable (the contribution):** small `f_a, f_v` projections; `M = cos(f_a(ā), f_v(V))`; train
   MIL-max contrastive — high max-similarity on **matched** audio, low on **swapped/silent** audio.
4. **Select** localizing heads / drop sink columns if using the attention route.

---

## 3. The controls — the part that turns a picture into a result

Run all of these; a map that doesn't pass them is "visual saliency in an audio costume."

- **Vision-only / audio-removed floor** — strip the audio; our map MUST beat it (and a trivial
  center-Gaussian + generic-saliency map). If it ties, it's saliency. (Oya ACCV'20
  `arxiv.org/abs/2007.05722`)
- **Audio-swap (our clips)** — same frame, different clip's audio; the map must move to the swapped
  object or collapse. Report matched→swapped **Δ(cIoU / pointing-precision)** + a permutation test.
  (`arxiv.org/abs/2410.01020`)
- **Silence / noise / offscreen negatives** — map must deactivate; score positive∪negative with
  **AP + max-F1** (SLAVC `arxiv.org/abs/2209.09634`) and **FLOC/FAUC** (`github.com/xavijuanola/vssl_eval`).
- **IS3+ implausible pairs** — diffusion scenes that decorrelate saliency from the sounding object;
  the map must follow the audio's object. (`arxiv.org/abs/2508.21761`)
- **Hygiene:** do NOT early-stop or threshold-tune on the annotated cIoU set (SLAVC flaw #1); avoid
  Adaptive cIoU if you also want to claim threshold-free readiness.

**Metrics:** cIoU@0.5 + AUC (Senocak/LVS) and the **pointing game** (argmax-in-box, scale-robust);
mIoU + F-score if producing masks (AVSBench).

**Benchmarks (spatial GT):** VGG-SS (~5k boxes, 220 classes), AVSBench S4/MS3/AVSS (masks),
Flickr-SoundNet (250 test), IS3+ (synthetic + negatives). **AVE is temporal-only — not a WHERE
benchmark.**

---

## 4. The honest niche & risk

- Both shipping AV-LLM grounding systems on Omni — supervised **Meerkat** (coordinate decoding,
  `arxiv.org/abs/2407.01851`) and training-free **Qwen2.5-Omni meta-reasoning SSL** (text reasoning,
  not attention, `arxiv.org/abs/2604.06824`) — **avoid internal-attention readout.** Nobody has shown
  a frozen Omni's attention/tokens give a reliable sound-localization heatmap.
- That absence is **our opening** (novelty) *and* a warning (it's probably noisy without
  gradient-weighting, sink-cleaning, head-selection, and contrastive audio training). The deliverable
  that makes it a paper is not the map — it's the map **plus the control battery proving it listens.**

## Recommended next step

Build the **free cosine-similarity + attention×gradient localizer on TRUE matched clips, with the
audio-swap control wired in** (no training). One job runs the whole diagnosis: does Qwen localize the
sound at all, does it beat the vision-only/saliency floor, and does the map *move* when we swap the
audio. If yes → train the alignment bottleneck to sharpen it. If no → the trainable projection is
required to make audio/visual tokens comparable. Diagnose before building, as before.
