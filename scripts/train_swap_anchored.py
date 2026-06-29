#!/usr/bin/env python
"""Collapse-resistant anchored swap-DPO (the rebuild). See docs/research/dpo-collapse-and-fixes.md.

Fixes the degenerate-"no"/garbage collapse of train_swap.py by adding:
  1. the mDPO chosen anchor (pins the chosen log-prob >= the frozen base), and
  2. an explicit KL-to-base on GENERAL (non-swap) prompts -- a mix of the matched MCQ and
     yes/no probes -- so the always-on adapter stays identity where it should.
Plus the monitoring we lacked: a per-step yes-fraction probe (the cheap collapse detector),
and periodic checkpoints so model selection happens on the held-out BENCHMARKS (run
run_bottleneck_eval.sh on the saved checkpoints), never on the in-distribution proxy.

  python scripts/train_swap_anchored.py --pairs 300 --epochs 2 --accum 4   # Stage 1 (unconditional)

Prompt-aware (FiLM) Stage 2 -- warm-start the query-conditioned bottleneck from a Stage-1
checkpoint and learn question-routed grounding (HEAR vs SEE on the SAME swapped clip forces the
gate to route by question). See docs/research/film-multistage-recipe.md and
docs/research/query-conditioned-bottleneck.md:

  python scripts/train_swap_anchored.py --film \
      --init-from runs/anchored_qwen3-omni/bottleneck_step60.pt \
      --pairs 400 --epochs 3 --accum 4 --see-frac 1.0 --warmup-steps 80 \
      --warmup-frac 0.1 --lr 3e-5 --lam-anchor 1.5 --lam-kl 2 --beta-kl 0.05 \
      --lam-gate 0.05 --gate-target 0.6
"""
from __future__ import annotations

import argparse
import collections
import os
import random

import torch

from rlvib.data import ave
from rlvib.data.pairs import make_swap_examples
from rlvib.models import get_model
from rlvib.models.bottleneck import (
    FiLMVariationalBottleneck,
    VariationalBottleneck,
    attach_bottlenecks,
    question_embedding,
    set_bypass,
    set_condition,
)
from rlvib.train.dpo import anchored_dpo_step, answer_logp_vec, letter_id


# Diverse anchor prompts so the KL-to-base covers the behaviors CMM/AVHBench actually
# test -- audio-presence AND visual-presence (the CMM hallucination axis) + MCQ + open-
# ended -- not just the matched MCQ. Visual-presence with an ABSENT category is the key
# addition: it anchors "don't claim to see things that aren't there" to the frozen base.
# With FiLM, this same set also spans the HEAR/SEE query axis the gate now conditions on.
_ANCHOR_KINDS = ["mcq", "hear", "see", "describe", "whatav"]


def _anchor_msg(model, it, cats, rng):
    """A diverse general (non-swap) prompt where the adapter should match the frozen base.
    Returns (messages, prompt_text); prompt_text is the FiLM conditioning string (no-op otherwise)."""
    v = it["video_path"]
    kind = rng.choice(_ANCHOR_KINDS)
    if kind == "mcq":
        mcq = ave.make_mcq(it["category"], cats, rng=rng)
        text = ave.format_mcq(mcq["question"], mcq["options"])
    elif kind in ("hear", "see"):
        present = rng.random() < 0.5
        cat = it["category"] if present else rng.choice([c for c in cats if c != it["category"]])
        verb = "HEAR the sound of" if kind == "hear" else "SEE"
        text = f"Do you {verb} {cat} in this clip? Answer yes or no."
    elif kind == "describe":
        text = "Describe what happens in this clip in one sentence."
    else:
        text = "What do you see and hear in this clip?"
    return model.message(video=v, prompt=text), text


def _yesno_probe(model, items, cats, rng):
    probe = []
    for it in items:
        if rng.random() < 0.5:
            cat, gold = it["category"], "yes"
        else:
            cat, gold = rng.choice([c for c in cats if c != it["category"]]), "no"
        ptext = f"Do you HEAR the sound of {cat} in this clip? Answer yes or no."
        msg = model.message(video=it["video_path"], prompt=ptext)
        probe.append({"messages": msg, "gold": gold, "prompt": ptext})
    return probe


@torch.no_grad()
def _frac_yes(model, probe, bns=None):
    """Fraction the model answers 'yes' (the cheap collapse detector) + balanced accuracy.
    Sets the FiLM condition per item when `bns` carries a q_proj (no-op otherwise)."""
    yes_ids = [letter_id(model, t) for t in ("yes", "Yes", " yes")]
    no_ids = [letter_id(model, t) for t in ("no", "No", " no")]
    nyes = nacc = 0
    for ex in probe:
        if bns is not None and "q_proj" in bns:
            set_condition(bns, question_embedding(model, ex["prompt"]))
        lp = answer_logp_vec(model, ex["messages"])
        pred = "yes" if max(float(lp[i]) for i in yes_ids) > max(float(lp[i]) for i in no_ids) else "no"
        nyes += pred == "yes"
        nacc += pred == ex["gold"]
    k = max(1, len(probe))
    return nyes / k, nacc / k


@torch.no_grad()
def _routing_probe(model, bns, records):
    """FiLM diagnostic: forward each swapped clip under HEAR and SEE, report the mean relative
    edit ||edit||/||x|| per modality x question-type + the gate. Success = audio edited MORE on
    HEAR than SEE (d_audio>0) and vision MORE on SEE than HEAR (d_vision>0)."""
    agg = collections.defaultdict(list)
    qs = {"hear": "Which event do you HEAR in this clip?",
          "see": "Which event do you SEE in this clip?"}
    for r in records:
        for qtype, q in qs.items():
            set_condition(bns, question_embedding(model, q))
            msg = model.message(video=r["video_path"], prompt=ave.format_mcq(q, r["options"]))
            answer_logp_vec(model, msg)                    # fires the hooks -> fills last_* per bn
            for mod in ("audio", "vision"):
                b = bns[mod]
                rel = (b.last_residual_per_token / (b.last_input_norm_per_token + 1e-6)).mean()
                agg[(mod, qtype)].append(float(rel))
    avg = lambda key: (sum(agg[key]) / len(agg[key]) if agg[key] else float("nan"))  # noqa: E731
    ah, asee = avg(("audio", "hear")), avg(("audio", "see"))
    vh, vs = avg(("vision", "hear")), avg(("vision", "see"))
    return {"rel_a|H": ah, "rel_a|S": asee, "rel_v|H": vh, "rel_v|S": vs,
            "d_audio": ah - asee, "d_vision": vs - vh}


def _set_core_frozen(bns, frozen: bool) -> None:
    """Freeze/unfreeze the VIB core (enc/to_mu/to_logvar/out); FiLM film/gate + q_proj always train."""
    for name, b in bns.items():
        if name == "q_proj":
            continue
        for sub in ("enc", "to_mu", "to_logvar", "out"):
            for p in getattr(b, sub).parameters():
                p.requires_grad_(not frozen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="swap pairs per optimizer step")
    ap.add_argument("--anchor-batch", type=int, default=4, help="general anchor prompts per step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--lam-anchor", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--lam-kl", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--swap-dir", default="data/AVE/swapped")
    ap.add_argument("--save-dir", default=None, help="default: runs/anchored_<model>")
    ap.add_argument("--seed", type=int, default=0, help="training seed (data order + init) for repeats")
    ap.add_argument("--normalize-input", action="store_true",
                    help="LayerNorm the VIB input (auto-on for videollama2's massive activations)")
    # --- prompt-aware (FiLM) Stage-2 options (no effect without --film) ---
    ap.add_argument("--film", action="store_true",
                    help="train the query-conditioned FiLM bottleneck (Stage 2) instead of the "
                         "unconditional VIB; pair with --init-from to warm-start from Stage 1.")
    ap.add_argument("--init-from", default=None,
                    help="warm-start the VIB core from a Stage-1 checkpoint (FiLM/gate keep their "
                         "identity init; non-strict load).")
    ap.add_argument("--see-frac", type=float, default=1.0,
                    help="[FiLM] prob. of adding a matched SEE example per swapped clip (1.0 = "
                         "balanced HEAR/SEE -- the contrast that forces the gate to route; 0 = HEAR only).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="[FiLM] freeze the VIB core for the first N steps (Phase 2a: train only "
                         "FiLM/gate/q_proj), then unfreeze (Phase 2b). 0 = joint from the start.")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="[FiLM] linear LR warmup over this fraction of total steps (0 = constant LR).")
    ap.add_argument("--lam-gate", type=float, default=0.0,
                    help="[FiLM] gate-usage hinge weight: push the question-relevant modality's gate "
                         "above --gate-target (audio on HEAR, vision on SEE). 0 = off (rely on SEE data).")
    ap.add_argument("--gate-target", type=float, default=0.6, help="[FiLM] target gate for --lam-gate.")
    args = ap.parse_args()
    args.save_dir = args.save_dir or f"runs/anchored_{args.model}"
    torch.manual_seed(args.seed)
    # massive-activation backbones (VideoLLaMA2) need the scale-invariant VIB input;
    # Qwen-Omni (normal scale) stays off so its results are unchanged.
    nin = args.normalize_input or args.model == "videollama2"

    m = get_model(args.model)
    cls = FiLMVariationalBottleneck if args.film else VariationalBottleneck
    bns, handles = attach_bottlenecks(m, cls=cls, normalize_input=nin)
    if args.init_from:                                 # warm-start the core (Stage 1 -> Stage 2)
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = bns.load_state_dict(ck["state_dict"], strict=False)
        print(f"init-from {args.init_from}: loaded Stage-1 core "
              f"({len(unexpected)} unused keys; {len(missing)} new FiLM/gate/q_proj params kept at init)",
              flush=True)
    if args.film and args.warmup_steps > 0:
        _set_core_frozen(bns, True)                    # Phase 2a: train only FiLM/gate/q_proj
        print(f"[FiLM] Phase 2a: VIB core frozen for the first {args.warmup_steps} steps", flush=True)
    opt = torch.optim.AdamW(bns.parameters(), lr=args.lr)   # all params; frozen ones get no grad
    os.makedirs(args.save_dir, exist_ok=True)

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(args.seed)
    rng.shuffle(items)
    swap = make_swap_examples(items, args.pairs, args.swap_dir, cats, rng=rng)
    anchor_items = items[:400]
    probe = _yesno_probe(m, items[400:440], cats, random.Random(1))
    routing_records = list(swap[: min(8, len(swap))]) if args.film else []   # fixed FiLM diagnostic set
    print(f"swap pairs: {len(swap)} | anchor pool: {len(anchor_items)} | probe: {len(probe)}"
          + (f" | film (see-frac={args.see_frac})" if args.film else ""), flush=True)

    steps_per_epoch = max(1, len(swap) // args.accum)
    total_steps = args.epochs * steps_per_epoch
    sched = None
    if args.warmup_frac > 0:                            # linear LR warmup -> constant
        warmup_n = max(1, int(args.warmup_frac * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_n))

    set_bypass(bns, True)  # true frozen base
    bns.eval()
    base_yes, base_acc = _frac_yes(m, probe, bns)
    bns.train()
    set_bypass(bns, False)
    print(f"[probe base] frac_yes={base_yes:.2f} acc={base_acc:.2f}  (collapse = frac_yes -> 0 or 1)",
          flush=True)

    step = 0
    unfrozen = not (args.film and args.warmup_steps > 0)
    for epoch in range(args.epochs):
        rng.shuffle(swap)
        for i in range(0, len(swap) - args.accum + 1, args.accum):
            sb = []
            for r in swap[i:i + args.accum]:
                sb.append({"messages": m.message(video=r["video_path"],
                                                 prompt=ave.format_mcq(r["question"], r["options"])),
                           "prompt": r["question"], "qtype": "hear",
                           "chosen_letter": r["audio_letter"], "rejected_letter": r["visual_letter"]})
                if args.film and (args.see_frac >= 1.0 or rng.random() < args.see_frac):
                    see = ave.make_see_mcq(r["audio_event"], r["visual_event"], cats, rng=rng)
                    sb.append({"messages": m.message(video=r["video_path"],
                                                     prompt=ave.format_mcq(see["question"], see["options"])),
                               "prompt": see["question"], "qtype": "see",
                               "chosen_letter": see["visual_letter"], "rejected_letter": see["audio_letter"]})
            ab = []
            for _ in range(args.anchor_batch):
                msg, ptext = _anchor_msg(m, rng.choice(anchor_items), cats, rng)
                ab.append({"messages": msg, "prompt": ptext})
            mt = anchored_dpo_step(m, bns, opt, sb, ab, beta=args.beta, beta_kl=args.beta_kl,
                                   lam_anchor=args.lam_anchor, delta=args.delta, lam_kl=args.lam_kl,
                                   lam_gate=args.lam_gate, gate_target=args.gate_target)
            if sched is not None:
                sched.step()
            step += 1
            if not unfrozen and step >= args.warmup_steps:    # Phase 2a -> 2b boundary
                _set_core_frozen(bns, False)
                unfrozen = True
                print(f"[FiLM] Phase 2b: VIB core unfrozen at step {step}", flush=True)
            print(f"epoch {epoch} step {step}: "
                  + "  ".join(f"{k}={v:+.4f}" for k, v in mt.items()), flush=True)
            if step % args.eval_every == 0:
                bns.eval()
                fy, ac = _frac_yes(m, probe, bns)
                rmsg = ""
                if args.film:
                    rp = _routing_probe(m, bns, routing_records)
                    rflag = "  <-- ROUTING FAIL (d_audio<=0)" if rp["d_audio"] <= 0 else ""
                    rmsg = "  [route " + " ".join(f"{k}={v:+.3f}" for k, v in rp.items()) + "]" + rflag
                bns.train()
                ckpt = os.path.join(args.save_dir, f"bottleneck_step{step}.pt")
                torch.save({"state_dict": bns.state_dict(), "dim": m.hidden_dim,
                            "cls": cls.__name__, "model": args.model, "normalize_input": nin,
                            "cond_dim": (bns["audio"].cond_dim if args.film else None)}, ckpt)
                flag = "  <-- COLLAPSE ALARM" if (fy < 0.15 or fy > 0.85) else ""
                print(f"[probe step {step}] frac_yes={fy:.2f} (base {base_yes:.2f}) acc={ac:.2f}  "
                      f"saved {os.path.basename(ckpt)}{flag}{rmsg}", flush=True)
                if fy < 0.05 or fy > 0.95:
                    print("frac_yes collapsed past 0.05/0.95 -> stopping early.", flush=True)
                    for h in handles:
                        h.remove()
                    return 0

    for h in handles:
        h.remove()
    print("=== anchored training done ===  select a checkpoint with run_bottleneck_eval.sh", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
