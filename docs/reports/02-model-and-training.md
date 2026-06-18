# RLVIB — the model and its training procedures (reference)

A single place that explains **what we are training** and **every training procedure run on it**,
in order, with the one currently held as best. Companion to
[`01-anchored-swap-dpo.md`](01-anchored-swap-dpo.md) (the result writeup) and
[`../research/dpo-collapse-and-fixes.md`](../research/dpo-collapse-and-fixes.md) (the cited diagnosis).

---

## 1. The model

We do **not** train an AV-LLM from scratch. We take a **frozen** audio-visual LLM and insert a tiny
trainable **variational information bottleneck (VIB)** on its per-modality adapter outputs. Only the
bottleneck learns; the whole base model stays frozen.

```
  video ─▶ [vision encoder] ─▶ visual.merger ──hook──▶ ⟦ VIB_vision ⟧ ──┐
                                                                         ├─▶ Thinker (frozen LLM) ─▶ answer logits
  audio ─▶ [audio encoder ] ─▶ audio_tower.proj2 ─hook─▶ ⟦ VIB_audio  ⟧ ──┘
        └──────────── all frozen ────────────┘        only the two VIBs train
```

**Base models (one comparison arm each).** Same VIB design attaches to all three via
`model.adapter_modules()`; only the dim and the eval/training readiness differ.

| Model | wrapper | adapter attach points | hidden dim | training |
|---|---|---|---|---|
| **Qwen3-Omni** (primary) | `qwen3_omni.py` | Thinker `audio_tower.proj2` / `visual.merger` | 2048 | ✅ |
| Qwen2.5-Omni | `qwen25_omni.py` | `audio_tower.proj` / `visual.merger` | 3584 | ✅ (parked) |
| VideoLLaMA2.1-7B-AV | `videollama2.py` | `mm_projector_a` / `mm_projector` | 3584 | eval-only* |

\*VideoLLaMA2 generates via `mm_infer`; it has no differentiable answer-logits path yet, so the DPO
step can't run on it. The bottleneck *attaches* and eval works.

**The bottleneck** (`models/bottleneck.py::VariationalBottleneck`), attached by a forward hook on each
adapter, so its input `x` is the adapter's output token sequence `(T, dim)`:

```
h      = GELU(enc(x))
mu     = to_mu(h)
logvar = to_logvar(h).clamp(-8, 8)
z      = mu + eps · exp(½·logvar)        # eps~N(0,I) when training; z = mu at eval
y      = x + out(z)                       # out is ZERO-INIT  ->  y = x at init (identity)
last_kl = mean( -½ (1 + logvar - mu² - exp(logvar)) )   # the IB "rate" term, KL(q(z|x) ‖ N(0,I))
```

Two key properties we exploit everywhere:
- **Zero-init output** ⇒ attaching the untrained bottleneck changes nothing; the model *is* the base.
- **`bypass` flag** ⇒ `y = x` on demand. Flipping bypass on recovers the **exact frozen base** with no
  second model in memory — this gives us the DPO *reference* log-probs and the KL-to-base *target* for free.

Only the bottleneck params (`enc, to_mu, to_logvar, out`, per modality) get gradients; the base is
`requires_grad_(False)`.

---

## 2. The training signal — modality-conditional (audio-swap) preference

Data: **AVE** clips (28 audio-visual event categories). For a clip we build a multiple-choice question
and a **swapped** copy of the clip whose audio is replaced by another category's audio. On that swapped
clip:
- **chosen** letter = the **heard** (audio-consistent) event,
- **rejected** letter = the **seen** (video-consistent) event.

Both are scored on the *same* swapped clip, so preferring `chosen` *requires using the audio* — that is
the grounding behavior we want to install. Notation used below (at the answer position):
`cp, rp` = policy log-prob of chosen/rejected letter (bottleneck **on**); `cr, rr` = reference log-prob
(bottleneck **bypassed** = frozen base); `KL_VIB` = the bottleneck rate term.

---

## 3. The training procedures, in order

### v0 — plain swap-DPO (latent-KL only)  ❌ collapsed
```
L = −log σ( β·[(cp − cr) − (rp − rr)] )  +  β_kl · KL_VIB
```
The *only* regularizer is `KL_VIB` on the **latent**; nothing anchors the **output** to the base.
Selected on an in-distribution `heard_rate`.
**Result:** `heard_rate` 0.70→1.00 while the model **catastrophically collapsed** — CMM-PA 0.953→**0.007**,
HR→~0.99 (constant "no"), DAVE 0.380→0.18. Diagnosed as **likelihood displacement** (the DPO margin is a
log-sigmoid of a *difference*, so it can be won by pushing **both** answer log-probs down) with no output
anchor, on an always-on adapter under an input-blind loss. Full cite trail in the research memo.

### v1 — anchored swap-DPO, λ_kl = 1 (narrow anchor)  ⚠️ drifts
Add the two anchors v0 lacked:
```
L = −log σ( β·[(cp−cr) − (rp−rr)] )                      # swap-DPO
    + λ_anchor · ( −log σ( β·(cp − cr) − δ ) )           # mDPO chosen anchor: pin chosen ≥ base
    + λ_kl     · KL( p_base(·|x) ‖ p_policy(·|x) )       # KL-to-base on GENERAL inputs x
    + β_kl     · KL_VIB
```
Hyperparams: β=0.1, β_kl=0.01, λ_anchor=1, δ=0, **λ_kl=1**, lr=5e-5, 300 pairs, 2 epochs → 150 steps,
AdamW (bottleneck only). General anchor inputs `x` = AVE matched-MCQ + audio-presence yes/no.
**Result:** no catastrophe, but a **capability/grounding tradeoff** — CMM-HR drifts 0.780→**0.247** by
step150; only step30 stays clean and its AVHBench gain is within noise. The big step150 AVHBench (0.763)
was a **yes-bias artifact**, not grounding. → the anchor was too **narrow** (never saw CMM-style
visual-hallucination inputs) and too **weak**.

### v2 — anchored swap-DPO, λ_kl = 4 (stronger anchor)  ✅ first clean gain  ← **CURRENTLY HELD**
Identical to v1 but **λ_kl = 4** (stronger output anchor).
**Result:** drift controlled; step30/60/90 clear the guards. **step60 is the first clean(-ish) win:**
AVHBench 0.643→**0.680 (+3.7)** with CMM near base (PA 0.933, HR 0.740) — both within tolerance. Late
steps (120/150) degrade, so we **select mid-training**. Small real cost for a small real gain (modest at
n=300; wants confirmation).

### v3 — broad anchor, λ_kl = 2 (broadened coverage)  ⏳ in flight
Same as v2 but the general anchor inputs are **broadened** (`_anchor_msg`) to span **MCQ + audio-presence
+ VISUAL-presence (asking about *absent* categories) + open-ended** — so the KL-to-base now protects the
*visual-hallucination* behavior CMM actually tests, not just AVE audio yes/no. λ_kl=2, `EXP=broad`.
**Status:** training/selection in progress. Tests whether **coverage** beats raw **strength** (v2).

---

## 4. Results so far (Qwen3-Omni, n=300)

Read each row as: **AVHBench** = the grounding gain we want ↑; **CMM_PA / CMM_HR** = capability guards
that must stay near base (PA≥0.90, HR≥0.70) — *one* of them collapsing while AVHBench rises = a bias
artifact, not grounding; **DAVE** = sanity guard.

| run | best ckpt | AVHBench | CMM_PA | CMM_HR | verdict |
|---|---|---|---|---|---|
| base | — | 0.643 | 0.953 | 0.780 | reference |
| v0 plain DPO | — | — | **0.007** | ~0.99 | catastrophic collapse (constant "no") |
| v1 λ_kl=1 | step30 | 0.657 | 0.960 | 0.780 | clean but gain ≈ noise; later steps crater HR→0.247 |
| **v2 λ_kl=4** | **step60** | **0.680** | **0.933** | **0.740** | **first clean gain (+3.7), capability within tolerance** |
| v3 broad | — | — | — | — | in flight |

---

## 5. Current status — what is held

- **Held recipe:** **v2** — anchored swap-DPO with the chosen anchor + KL-to-base, **λ_kl = 4**,
  **select mid-training (step60)**. It is the only procedure so far that lifts AVHBench while keeping
  both CMM axes within tolerance.
- **Held checkpoint:** `runs/anchored_qwen3-omni/bottleneck_step60.pt` (AVHBench 0.680, PA 0.933, HR 0.740).
- **In flight:** **v3 broad** (does broadened coverage beat strength?). Then, if a recipe gives a
  confirmed clean gain, replicate to **Qwen2.5-Omni** (trained, selection parked) and implement the
  **VideoLLaMA2** training-forward to bring in the third arm.

**Monitoring/selection discipline (so collapse is caught early):** per-step `frac_yes` probe +
`chosen_minus_ref` (anchor floor, ≥0) + `gen_kl`; model selection on the **held-out benchmarks** with the
PA & HR guards — never on the in-distribution proxy.
