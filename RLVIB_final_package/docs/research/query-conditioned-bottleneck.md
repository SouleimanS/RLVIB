# Query-conditioned bottlenecks — will conditioning the VIB on the prompt enhance the model?

> Date: 2026-06-18 · Status: design memo + literature review (decision-oriented).
> Method: 5-angle deep-research fan-out (conditioning mechanisms · conditional-IB theory ·
> query-conditioning→grounding evidence · conditioning×DPO-collapse dynamics · IB-rate
> abstention + novelty), then a **3-vote adversarial verification pass** on 18 load-bearing
> claims (all 18 resolve to real papers; corrections folded in — §9).
> Extends `ib-rl-method-and-framing.md`, `dpo-collapse-and-fixes.md`,
> `../reports/02-model-and-training.md` (the unconditional-rewrite finding that motivates this).

## 0. TL;DR — decision

- **GO, but staged and controlled — not a wholesale jump.** Conditioning the bottleneck on the
  prompt/query is well-motivated and well-supported, *but* the literature's dominant lesson is
  that query-conditioning reliably buys **accuracy** while frequently **not** buying **faithful
  grounding** (it can learn a question→answer shortcut). So adopt it as a measured ablation under
  the project's existing faithfulness controls, not as a free win.
- **Recommended scope: (1) query-text-only conditioning of the existing per-modality VIBs.**
  Smallest change; keeps per-modality separation + interpretability; lowest collapse risk; best
  direct precedent (InstructBLIP, QA-TIGER, FiLM, QG-VTC — all frozen-backbone). Defer (2)
  query+cross-modal and (3) joint AV+query until (1) shows a *faithful* gain.
- **Recommended mechanism: FiLM.** Feed a pooled query embedding `q` into each VIB and FiLM-modulate
  its hidden activations. Because `out` is zero-init, **identity-at-init and the `bypass`→exact-base
  reference are preserved for free** — so the anchored-DPO machinery is untouched (§5, §7).
- **The decisive risk is training dynamics, not architecture.** A query-conditioned always-on adapter
  is *net-destabilizing* under DPO (adds a query→answer-shortcut path and, if you use gating/MoE, a
  routing-collapse path) on top of the displacement collapse we already hit. Mitigations are concrete
  and evidence-backed (§6): keep anchored-DPO, **broaden the KL-to-base anchor to span the query
  distribution**, prefer FiLM over discrete routing, raise `β_kl`, and **gate model selection on
  faithfulness (AVHBench + CMM-HR + audio-swap), never on accuracy alone.**
- **Novelty: the lane is open** (§8). No published work conditions a *VIB's* `μ/σ` on the query;
  doing it on the AV fusion code of a *frozen* MLLM, RL-trained, is a genuine increment. Highest
  scoop risk: **Vittle** (VIB-in-MLLM, but fine-tuned, vision-text, *unconditional*).

---

## 1. Why ask this now (the motivating finding)

The held v3 VIB grounds by rewriting the **vision** token stream **unconditionally**: ~59% relative
edit, diffuse over all ~1.6k tokens, *near-identical across clips and categories*
(`../reports/02-model-and-training.md` §7). That is a query-blind global re-projection — the
"trust audio over video" decision is made by the **frozen LLM** on reshaped vision inputs, not by
the bottleneck. If the edit is the same regardless of what is asked, the obvious lever is to let the
bottleneck **read the question** and compress AV bits by *relevance to the query*. This memo asks
whether that helps, how to wire it without breaking the frozen-base/anchored-DPO design, and at what
risk.

---

## 2. Does query-conditioning help? (architecture + grounding evidence)

**Yes for accuracy, under a frozen backbone — the mechanism is proven.**

- **InstructBLIP** feeds the instruction into the Q-Former (frozen ViT + frozen LLM); ablating the
  instruction-conditioning *hurts* — ~5–7% on ScienceQA, ~8–15% on iVQA — i.e. conditioning the
  small module on the query matters most on reasoning-heavy items where different questions need
  different visual evidence. [arXiv:2305.06500, NeurIPS'23] This is the closest precedent to our
  setup (small trainable module, both backbones frozen). By contrast **BLIP-2**'s 32 Q-Former
  queries are *query-agnostic* at feature extraction — InstructBLIP exists precisely to fix that.
  [arXiv:2301.12597]
- **QA-TIGER** routes the question into a mixture of Gaussian temporal experts for **audio-visual**
  QA → SOTA on MUSIC-AVQA (~77.6%). [arXiv:2503.04459, CVPR'25] Direct in-domain evidence that
  question-conditioned bottlenecking helps AVQA.
- **FiLM** (per-channel affine modulation from a question encoder) reaches 97.7% on CLEVR — the
  canonical, cheapest query-conditioning primitive. [arXiv:1709.07871, AAAI'18]
- **QG-VTC** scores question→visual-token relevance with a small trainable text-encoder+FFN over a
  **frozen** backbone and prunes tokens while retaining VQA accuracy. [arXiv:2504.00654] Shows the
  conditioning signal can be a light add-on, not a backbone change. (Token-pruning / efficiency
  cousins: **SCRIPT** training-free query-conditioned pruning [arXiv:2512.01949]; query-conditioned
  hypernetwork-LoRA for frozen LLMs — **Drag-and-Drop** [arXiv:2506.16406], **Zhyper**
  [arXiv:2510.19733].)

**The load-bearing caveat — accuracy ≠ faithful grounding (read this twice).**

- "Visual grounding methods for VQA are **working for the wrong reasons**": question-conditioned
  attention supervision improves the bias split via a **regularization** effect — *random/nonsensical
  cues give similar gains*. [arXiv:2004.05704, ACL'20]
- **FPVG** shows many models score high on *plausible* grounding (attend to the right region) yet low
  on *faithful* grounding (answer actually depends on it). [arXiv:2305.15015, EMNLP'23-F]
- The debiasing **see-saw**: reducing the language prior often *raises* visual bias; killing both at
  once needed causal intervention, not plain conditioning. [Counterfactual VQA, arXiv:2006.04315,
  CVPR'21 — ~+13 pts on VQA-CP v2 by subtracting the question's direct effect; PW-VQA arXiv:2305.19664]
- **iGVLM**'s ablation: removing the *static, unconditional* branch hurt **more** than removing the
  instruction-conditioned (AdaLN) branch → **keep the unconditional path and *add* conditioning;
  don't replace it.** [arXiv:2603.02748, Mar 2026]
- AVQA answer skew makes the shortcut easy: >90% "yes" on some MUSIC-AVQA templates
  [arXiv:2310.06238]; question-conditioned cycle-debiasing nonetheless gains **+9.32%** on the
  robustness split MUSIC-AVQA-R [Look-Listen-Answer, arXiv:2404.12020, NeurIPS'24].

**Implication:** conditioning will very likely move AVHBench/accuracy; that is *not* sufficient
evidence it grounds better — exactly the trap v1-narrow fell into (a yes-bias that lifted AVHBench
while CMM-HR cratered). Acceptance must be faithfulness-gated (§6).

---

## 3. What conditioning buys information-theoretically (and the catch)

- **Conditional Entropy Bottleneck**: replace the rate `I(Z;X)` with the **conditional rate
  `I(Z;X|Y)`** → keep only the information about X *not already explained by the conditioner*; this
  is the Minimum-Necessary-Information point, and it improves calibration/OOD robustness.
  [objective: Fischer, arXiv:2002.05379, *Entropy* 2020; robustness: arXiv:2002.05380] The
  multimodal instance — **conditional IB for sarcasm**, conditioning on the *other modality* — shows
  it suppresses unimodal shortcuts while keeping cross-modal info. [arXiv:2508.10644, AAAI'26]
- **The catch (3 caveats the memo must respect):**
  1. **Q is not Y.** All CEB/CIB work conditions on the *label* or an *auxiliary modality*, both
     known at train time. An exogenous, inference-varying **query Q** is formally distinct (a
     side-information-at-decoder / Wyner-Ziv flavour); the clean MNI reading of `I(Z;X|Q)` is not
     guaranteed. [inference from arXiv:physics/0004057, 2002.05379]
  2. **Conditioning can *add* shortcuts.** If Q correlates with a spurious cue (and VQA questions
     notoriously do), conditioning installs a *query-specific* shortcut rather than removing one.
     [synthesis: 2004.05704 + the VQA-bias line]
  3. **`I(Z;X|Q)` is noisier to estimate** than `I(Z;X)` (extra marginalization, higher variance).
     [Poole et al., arXiv:1905.06922; Molavipour et al., arXiv:1911.02277] Combined with the fact
     that our IB rate **barely bites today** (`β_kl=0.01`, mean-KL over 2048 dims ≈ 0.002), do **not**
     lean on the conditional rate as the abstention signal yet — treat abstention as a separate,
     later test (InfoRM's latent-geometry CSI may beat a scalar rate [arXiv:2402.09345]; cf.
     Distance-Aware Bottleneck rate-as-OOD [arXiv:2406.10775, ICML'24]).

---

## 4. The three scopes — comparison & verdict

| | expressiveness | grounding evidence | collapse risk | interpretability / `bypass` | compute | verdict |
|---|---|---|---|---|---|---|
| **(1) query-text only** (per-modality VIB reads Q) | low–med | strongest (InstructBLIP, QA-TIGER, FiLM, QG-VTC) | **lowest** | **preserved** (per-modality KL; clean bypass) | + tiny | ✅ **do first** |
| **(2) query + cross-modal** (audio-VIB sees video+Q) | med–high | CEB/sarcasm theory; true conditional fusion-IB | higher | weakened (per-modality separation breaks) | + | ⏳ v2 if (1) grounds |
| **(3) joint AV+Q** (one bottleneck over A+V+Q) | highest | best for synergy in principle | **highest** (routing + shortcut) | lost; bypass/ref harder | ++ | 🚫 not now |

**Why (1) first:** it is the minimal, reversible change that *directly tests the motivating finding*
(does an input-dependent edit beat the unconditional rewrite?), preserves every property the
anchored-DPO design exploits, and yields a clean paper-ready ablation (**conditional vs unconditional
VIB**). Scopes (2)/(3) trade the per-modality interpretability and the free exact-base reference for
synergy we have no evidence we currently need (audio is barely edited today), at strictly higher
collapse risk — revisit only if (1) plateaus.

---

## 5. Feasibility — can a pre-LLM, hook-attached bottleneck even see the query?

**Not as written.** `attach_bottlenecks` (`models/bottleneck.py`) registers a `forward_hook` on each
adapter; the hook passes **only the adapter output** to `bn(output[0])`. The prompt tokens are
embedded *separately* and concatenated into `inputs_embeds` — they are **not** in the tensor the hook
sees. So `VariationalBottleneck.forward(x)` is query-blind by construction; conditioning requires
*explicitly supplying* a query vector to the bottleneck.

**The fix is small and preserves everything:**
- Compute one **pooled query embedding** `q` per sample (reuse the model's own *frozen* token
  embedding of the prompt → mean-pool; same space, ~zero new params), and stash it on the bottleneck
  objects before the model forward (`bottlenecks["vision"].q = bottlenecks["audio"].q = q`). The hook
  is unchanged; `forward` reads `self.q`.
- `forward(self, x)` falls back to **exactly today's behaviour when `self.q is None`** → backward
  compatible, and the unconditional checkpoint stays loadable.
- Because `out` is **zero-init**, `y = x` at init *regardless of the conditioning* → identity-at-init
  and `bypass`→exact-base both hold unchanged. **This is the key enabler: the anchored-DPO reference
  (bypass = frozen base) and the per-modality `last_kl` saliency probe keep working as-is.**

---

## 6. Interaction with the DPO collapse (the decisive section)

The evidence says input/query-conditioning is **net-destabilizing on balance** under preference
optimization — it adds new failure modes on top of the likelihood-displacement collapse we already
diagnosed (`dpo-collapse-and-fixes.md`). Manage all four:

1. **Query→answer shortcut (the #1 risk).** A conditioned module can learn question-phrasing→answer
   maps that bypass AV grounding — the dominant VQA failure (§2). *Mitigation:* keep the **audio-swap
   counterfactual** (the answer must follow the *audio*, not the question); raise **`β_kl`** so the
   bottleneck can't cheaply memorize Q→A (it barely bites today); add a **query-only control**
   (conditioning must *not* help when AV is ablated).
2. **Routing/dead-path collapse.** Discrete query-gating/MoE self-reinforces: under-visited query
   types starve of gradient → near-random edits (the MoE expert-collapse failure; standard fix =
   load-balancing loss [Shazeer'17]). *Mitigation:* **use FiLM, not routing** — dense conditioning
   updates the shared generator for every query, so there is no routing to collapse. (This is a
   concrete reason to prefer FiLM for scope 1.)
3. **Anchor coverage must grow with the conditioning space.** Offline DPO needs *global* coverage of
   the reference; its KL only bites where it has support [arXiv:2406.01462, NeurIPS'24]. A
   query-conditioned adapter enlarges the effective input space, so the **KL-to-base general-anchor
   set must now span the query distribution** (MCQ + audio-/visual-presence + open-ended). This is
   the *same* mechanism by which **v3-broad beat v1-narrow** — broaden it one more step, along the
   query axis.
4. **Selection on faithfulness, never accuracy.** Because conditioning buys accuracy without
   guaranteeing grounding (§2), keep the held-out gate exactly as v3: accept only if AVHBench rises
   **and CMM-HR stays ≥ base** (HR collapsing while AVHBench rises = the yes-bias artifact, not
   grounding), with CMM-PA ≥ 0.90 and DAVE ≥ 0.36; add a per-query-type `frac_yes` probe.

mDPO is the encouraging counter-signal: making preference *explicitly conditional on the modality*
(its conditional term + chosen-reward anchor) **stabilized** multimodal DPO (MMHalBench 2.28→2.96).
[arXiv:2406.11839] Our audio-swap term is already that conditional preference; query-conditioning is
complementary, provided the anchor/coverage discipline above holds.

---

## 7. Recommended design (snaps into `bottleneck.py`) + protocol

**Architecture — `QueryConditionedVIB` (scope 1, FiLM):** identical to `VariationalBottleneck` plus a
query-FiLM on the hidden activation; per modality; `q` set externally per batch.

```
# new: film = nn.Linear(dim_q, 2*hidden)            # generates (gamma, beta) from pooled q
def forward(self, x, q=None):                        # q optional -> unconditional fallback
    if self.bypass: return x
    h = self.act(self.enc(x.to(self.enc.weight.dtype)))
    if q is not None:
        gamma, beta = self.film(q).chunk(2, dim=-1)  # (B, hidden) each
        h = (1 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)   # FiLM over the token axis
    mu = self.to_mu(h); logvar = self.to_logvar(h).clamp(-8, 8)
    z  = mu + torch.randn_like(mu)*torch.exp(0.5*logvar) if self.training else mu
    self.last_kl = (-0.5*(1+logvar-mu.pow(2)-logvar.exp())).mean()
    return x + self.out(z).to(x.dtype)               # out zero-init -> identity at init, q or not
```

- `q` = mean-pooled **frozen** prompt-token embeddings (no new encoder); `dim_q = hidden_dim`.
- `out` stays zero-init ⇒ identity-at-init and `bypass`→base unchanged (§5). Keep **per-modality**
  (each VIB gets its modality tokens + the shared `q`); `last_kl`/saliency probe unchanged.
- Wire `q` in `attach_bottlenecks`/the train loop by setting `bn.q` before each forward (the hook
  body needs no change since it calls `bn(output[0])` and `forward` reads `self.q`; or pass `q`
  through a tiny closure tweak).

**Training (reuse the held v3 recipe, with the §6 additions):**
- Loss unchanged: anchored swap-DPO (swap-DPO + mDPO chosen anchor + KL-to-base + `β_kl·KL_VIB`).
- **Broaden the KL-to-base anchor along the query axis** (the §6.3 extension of v3-broad).
- **Raise `β_kl`** (e.g. 0.01→0.1+) — real compression pressure, doubling as a shortcut guard (§6.1);
  this also tests the open `β_kl` question from `02-model-and-training.md` §7.
- Selection: faithfulness-gated, mid-training, per the held protocol (§6.4).

**The experiment that answers the question (one ablation):** train **unconditional VIB vs
query-conditioned VIB** under the *same* anchored recipe and *same* selection gate on Qwen3-Omni;
report AVHBench **and** CMM-PA/HR **and** the audio-swap heard-rate **and** DAVE. Conditioning "wins"
only if it lifts grounding faithfully (HR ≥ base) — not on AVHBench/accuracy alone.

---

## 8. Novelty / scoop

Query-conditioned **VIB** on the **AV fusion** code of a **frozen** MLLM, RL-trained, with
abstention read off the (conditional) rate — **no published work occupies this**. The components
exist apart: conditioning a *Q-Former* (InstructBLIP — not a VIB, no RL/IB); a VIB *inside an MLLM*
(**Vittle** — but **fine-tuned**, vision-text, **unconditional**, no abstention — *highest scoop
risk* [arXiv:2505.13946, NeurIPS'25]); conditional IB on the *other modality* (sarcasm CIB — not a
query, no LLM/RL [arXiv:2508.10644]); a VIB *probe of attention-head outputs* in a frozen VLM
(VIB-Probe — vision-text, unconditional [arXiv:2601.05547]). So conditioning is also a **novelty
increment**, not just an engineering tweak — and it sharpens, rather than overlaps, the existing
RLVIB positioning in `ib-rl-method-and-framing.md` §3.

---

## 9. Verification notes (3-vote adversarial pass; corrections applied)

All 18 spot-checked citations are **real**; none was refuted ≥2/3. Corrections folded in above:
- **QG-VTC** [2504.00654] benchmarks **VQA only** — its paper does **not** itself show query-guided
  pruning *degrading visual grounding*; that accuracy≠grounding point is carried by FPVG /
  "wrong-reasons" instead (not attributed to QG-VTC).
- **CEB**: the *objective* `I(Z;X|Y)` is in [2002.05379, *Entropy* 2020]; the *robustness/calibration*
  results are in the companion [2002.05380]. Cited separately.
- **Sarcasm conditional-IB** [2508.10644] venue is **AAAI 2026** (matches `ib-rl-method-and-framing.md`;
  an interim search had said 2025).
- **iGVLM** [2603.02748] is **March 2026** (arXiv `2603.*`), not 2025; **SCRIPT** [2512.01949] is
  **Dec 2025**; **Drag-and-Drop** [2506.16406] is **2025**.
- Number nuances softened: InstructBLIP ScienceQA-ablation drop is ~3–7% (backbone-dependent;
  iVQA ~8–15% holds); Counterfactual VQA is ~+13 pts on VQA-CP v2 (baseline ~37–39% → ~51%,
  backbone-dependent) — framed as a range, not a point.
- **VIB-Probe** [2601.05547] applies a VIB *as a probe of* attention-head outputs (not a VIB layer
  *over* the heads) — wording corrected.
- Medium-confidence / unverified-by-this-pass: the *direction* "conditional rate `I(Z;X|Q)` is a
  better per-query abstention signal" is **theoretically motivated but empirically unvalidated** in
  any source (treat as hypothesis, §3); QA-TIGER was SOTA *at publication* (later surpassed).

---

## Selected bibliography (grouped; ids verified this pass)

**Conditioning mechanisms (frozen-backbone)** — FiLM (AAAI'18, arXiv:1709.07871); BLIP-2 (ICML'23,
2301.12597); InstructBLIP (NeurIPS'23, 2305.06500); Flamingo (NeurIPS'22, 2204.14198); QG-VTC
(2504.00654); SCRIPT (2512.01949); QA-TIGER (CVPR'25, 2503.04459); Drag-and-Drop LLMs (2506.16406);
Zhyper (2510.19733); Object-aware APL for AVQA (AAAI'24, 2312.12816).

**Conditional-IB theory** — IB method (arXiv:physics/0004057); Deep VIB (ICLR'17, 1612.00410);
Conditional Entropy Bottleneck (Entropy'20, 2002.05379) + CEB-robustness (2002.05380); Predictive IB
(1910.10831); Variational bounds of MI (ICML'19, 1905.06922); CMI neural estimator (ICASSP'20,
1911.02277); conditional-IB sarcasm (AAAI'26, 2508.10644); Saxe et al. (2019, IB compression not
universal).

**Query-conditioning → grounding / bias** — Visual grounding "wrong reasons" (ACL'20, 2004.05704);
FPVG (EMNLP'23-F, 2305.15015); Counterfactual VQA (CVPR'21, 2006.04315); Cross-modality bias / PW-VQA
(2305.19664); Look-Listen-Answer (NeurIPS'24, 2404.12020); MUSIC-AVQA data bias (2310.06238); iGVLM
(2603.02748).

**Collapse / preference-opt dynamics** — Likelihood displacement (ICLR'25, 2410.08847); mDPO
(EMNLP'24, 2406.11839); squeezing effect (ICLR'25, 2407.10490); gradient imbalance (2502.20847);
online-data coverage / HyPO (NeurIPS'24, 2406.01462); OPA-DPO (CVPR'25, 2501.09695); MoE load
balancing (1701.06538); continual learning w/ hypernetworks (ICLR'20, 1906.00695).

**IB rate / abstention / novelty neighbours** — InfoRM (NeurIPS'24, 2402.09345); Distance-Aware
Bottleneck (ICML'24, 2406.10775); Vittle/VIBT (NeurIPS'25, 2505.13946); VIB-Probe (2601.05547); OMIB
(ICML'25, 2505.19996); CAL/Asymmetric-IB (2510.26289); CoMM (ICLR'25); OMD-Bench (2603.27187).
