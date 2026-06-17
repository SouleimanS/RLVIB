# Training-Data Sourcing & Counterfactual-Pair Plan

> Date: 2026-06-17 · Companion to `ib-rl-method-and-framing.md` and
> `grounding-audio-to-video.md`. From a 5-angle deep-research pass on training-data
> availability + pair construction.

## 0. TL;DR — the decision

- **Primary labeled-QA corpus:** **OmniInstruct-v1** (`m-a-p/OmniInstruct_v1`, ~84K train,
  inline audio+image) — one `load_dataset`, **no video scraping**.
- **Pilot:** **AVQA-R1-6K** (`harryhsing/AVQA-R1-6K`, 4.5K) — EchoInk-R1's exact set; fast sanity.
- **Raw-clip substrate to mint our *own* audio-dependent + counterfactual + mismatch pairs:**
  **VGGSound** (`Loie/VGGSound`, ~338 GB, sound source on-screen *by construction*, CC-BY 4.0)
  + **Kinetics-Sounds** (CVDF AWS tars, ~20–30 GB, 34 AV classes, no form).
- **Audio library for swap-in tracks:** AudioSet audio mirror (`agkphysics/AudioSet`).
- **Turnkey AV-mismatch data for abstention:** **AVID** (arXiv:2604.13593 — 39.4K inconsistency
  events, 8 categories) — confirm release on access.
- 🚫 **AVOID:** VALOR-1M / VAST-27M (YouTube-IDs / BaiduDisk — scraping pain); **EPIC-Kitchens &
  Ego4D (= the source of our DAVE eval → training on them leaks the test set)**; AVHBench-Align-FT
  *videos* (QA is in-repo but videos need VALOR+AudioCaps scraping → defer).
- **Held-out integrity:** VGGSound / Kinetics-Sounds / OmniInstruct don't overlap our eval sets
  (AVHBench ← VALOR/AudioCaps, CMM ← WebVid/AudioCaps, DAVE ← EPIC/Ego4D). Keep it that way; dedup
  by YouTube ID if paranoid.

## 1. Availability matrix

| Dataset | Access | Media shipped? | Size | Audio-salient | License | Role |
|---|---|---|---|---|---|---|
| **OmniInstruct-v1** | HF `m-a-p/OmniInstruct_v1` | ✅ inline audio+image | 84K train | mixed | research | **primary labeled QA** |
| **AVQA-R1-6K** | HF `harryhsing/AVQA-R1-6K` | ✅ inline | 4.5K | moderate | (check card) | pilot |
| **VGGSound** | HF `Loie/VGGSound` / yt-dlp | ✅ (mirror) / IDs | ~200K clips, 338 GB | **high** | CC-BY 4.0 | **build-own pairs** |
| **Kinetics-Sounds** | CVDF AWS tars | ✅ files | ~20–30 GB | high | research | secondary substrate |
| **AudioSet (audio)** | HF `agkphysics/AudioSet` | ✅ audio only | ~2.4 TB | mixed | CC-BY | swap-in audio library |
| **AVID** (mismatch) | arXiv:2604.13593 (check repo) | ? | 235 h / 39.4K events | — | ? | turnkey mismatch/abstention |
| AVHBench-Align-FT | GitHub (QA only) | ❌ videos need VALOR+AudioCaps | ~41–87K QA | — | unverified | defer (on-domain but scraping) |
| VALOR-1M / VAST-27M | YouTube IDs / BaiduDisk | ❌ | 1M / 27M | high | MIT(code) | **skip** (scraping) |
| EPIC-Kitchens / Ego4D | torrent / AWS+form | ✅ | 0.8–5 TB | high | NC + form | 🚫 **DAVE contamination** |

## 2. Pair construction (the recipe)

**Chosen** = intact A+V clip + correct answer (use real data).

**Rejected — tiered (strongest signal last):**
- **Tier A — audio-drop:** `ffmpeg -i v.mp4 -an -c:v copy out.mp4` → the audio-blind answer
  (OmniDPO mutes audio for its negatives).
- **Tier B — audio-swap (strongest):** the **Hungarian-matching** recipe from
  "Do AV-LLMs Really See and Hear?" [2604.02605]: embed all audio- and video-captions, solve a
  one-to-one assignment that *minimizes* similarity, keep pairs with cosine ≤ 0.5, then
  `ffmpeg -i v.mp4 -i mismatch.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest out.mp4`. Maximally
  semantic mismatch (AVHBench does this *randomly*; Hungarian makes it hard).
- **Tier C — abstention-mismatch:** on Tier-B clips, the *chosen* answer is "the audio and video
  are inconsistent — I can't answer from the AV evidence"; *rejected* = a hallucinated coherent
  answer. Label mismatch type with AVID's 8-category taxonomy.

**On-policy (adapt OPA-DPO to our setup):** since only the **bottleneck** is trainable, "on-policy"
means **rolling out the rejected responses from the current bottleneck checkpoint**, not a separate
model; re-roll every 1–2 DPO rounds. OPA-DPO hit SOTA with ~4.8K pairs — don't over-sample.

**Don't:** use text-only synthesized rejections — they miss the audio-grounding failure mode
(LISTEN, Interspeech'25). The rejected side must involve real audio/video manipulation.

## 3. Closest precedents (and our differentiation)

- **OmniDPO** (AAAI'26): audio-mute → rejected; per-modality conditional DPO.
- **MoD-DPO** (CVPR'26): invariance to irrelevant-modality corruption + sensitivity to relevant.
- **AVHBench**: random audio-swap mismatch (our Tier-B is the *Hungarian* upgrade).
- **LISTEN** (Interspeech'25): **frozen LLM + trainable adapter** + audio-negatives — the closest
  architecture to ours (but contrastive, not IB/RL).
- **OmniHalluc-L / MPRC** (Jun'26): **frozen-backbone** audio-negative probe for abstention
  calibration on Qwen2.5-Omni — the *inference-time* version of our Tier-C; we make it **trained**
  via the bottleneck + IB rate.
- **Our edge:** none couple these counterfactual pairs with an **explicit VIB on the fused
  bottleneck** + **IB-rate abstention** (see `ib-rl-method-and-framing.md` §3–4).

## 4. Concrete v0 data steps

1. `huggingface-cli download m-a-p/OmniInstruct_v1` (offline flags off, login node) → primary QA.
2. `huggingface-cli download Loie/VGGSound` (or yt-dlp from the CSV in a qsub job) → clip substrate.
   Filter with VGGSounder modality tags to the ~73% genuinely-visible clips.
3. (optional) pull **AVID** for ready-made mismatch/abstention data; AudioSet mirror for swap audio.
4. Build `rlvib/data/pairs.py`: ffmpeg swap/drop + Hungarian matching (CLAP or text-embed of
   captions) + on-policy rollout from the bottleneck → emits {prompt, chosen, rejected, tier}.

## 5. Caveats / verification

- **Dataset access verifies on download** — the HF ids above are concrete; confirm by pulling.
  Flagged-uncertain: AVHBench-train *videos* (need scraping), AVID/OmniVideo-100K release (too new),
  Pano-AVQA (access stale), VALOR-32K (BaiduDisk barrier).
- **Recipe sources:** 2604.02605 (Hungarian swap), AVID (2604.13593), LISTEN (2505.14518),
  OmniHalluc-L (2606.03614) are agent-sourced this pass; OmniDPO / MoD-DPO / OPA-DPO / mDPO /
  AVHBench were verified in earlier passes.
- **Licenses to check before publishing:** VGGSound CC-BY (ok), MUSIC-AVQA-v2 GPL-3.0 (careful),
  EPIC/Ego4D non-commercial + forms (and they're excluded anyway for contamination).
