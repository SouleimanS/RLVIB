#!/usr/bin/env python
"""GRPO over the VIB bottleneck on verifiable yes/no grounding items (docs/research/grpo-vib.md).

Reward is rule-based / verifiable (RLVR-style, no reward model): each training item is an
audio- or visual-presence question built from an AVE clip's category label -- "Do you HEAR the
sound of <cat>?" / "Do you SEE <cat>?" -- so the gold yes/no is known. grpo_step draws a group of
stochastic VIB passes per item, samples an answer, scores it (+1 correct / -1 hallucinate, plus 0
for an optional abstain token -> ternary, TruthRL-style), group-normalizes to advantages, and
updates only the bottleneck. The frozen backbone is the policy's body; exploration comes from the
VIB's z = mu + sigma*eps (watch `adv_std` -> 0 = zero-variance groups, `kl_vib` -> 0 = sampler
collapsing). Checkpoints land in runs/anchored_<model>_grpo so the eval tooling finds them:
  EXP=grpo bash scripts/eval_one.sh <model> <step>

  python scripts/train_grpo.py --model qwen3-omni --pairs 300 --epochs 2 --group 8
  ABSTAIN=unknown ...   # opt into the 3-way ternary reward (abstention as a third action)
"""
from __future__ import annotations

import argparse
import os
import random

import torch

from rlvib.data import ave
from rlvib.models import get_model
from rlvib.models.bottleneck import VariationalBottleneck, attach_bottlenecks, set_bypass
from rlvib.train.dpo import answer_logp_vec, grpo_step, letter_id


@torch.no_grad()
def _frac_yes(model, probe):
    """Fraction answered 'yes' (cheap collapse detector) + balanced accuracy on the probe."""
    yes_ids = [letter_id(model, t) for t in ("yes", "Yes", " yes")]
    no_ids = [letter_id(model, t) for t in ("no", "No", " no")]
    nyes = nacc = 0
    for ex in probe:
        lp = answer_logp_vec(model, ex["messages"])
        pred = "yes" if max(float(lp[i]) for i in yes_ids) > max(float(lp[i]) for i in no_ids) else "no"
        nyes += pred == "yes"
        nacc += pred == ex["gold"]
    k = max(1, len(probe))
    return nyes / k, nacc / k


def _yesno_item(model, it, cats, rng, suffix):
    """One verifiable yes/no presence item from an AVE clip (audio OR visual axis, balanced)."""
    axis = rng.choice(("hear", "see"))
    if rng.random() < 0.5:
        cat, gold = it["category"], "yes"
    else:
        cat, gold = rng.choice([c for c in cats if c != it["category"]]), "no"
    verb = "HEAR the sound of" if axis == "hear" else "SEE"
    msg = model.message(video=it["video_path"], prompt=f"Do you {verb} {cat} in this clip?{suffix}")
    return {"messages": msg, "gold": gold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--pairs", type=int, default=300, help="# yes/no training items")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1,
                    help="items per optimizer step (each item does --group forwards)")
    ap.add_argument("--group", type=int, default=8,
                    help="GRPO group size = stochastic forwards per item; LOWER (e.g. 4) if OOM")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta-kl", type=float, default=0.01, help="VIB compression-KL weight")
    ap.add_argument("--lam-ref", type=float, default=0.05, help="KL(base||policy) anchor weight")
    ap.add_argument("--abstain-token", default="",
                    help="enable 3-way ternary reward via this token id (e.g. 'unknown'); '' = 2-way")
    ap.add_argument("--r-correct", type=float, default=1.0)
    ap.add_argument("--r-abstain", type=float, default=0.0)
    ap.add_argument("--r-halluc", type=float, default=-1.0)
    ap.add_argument("--yn-suffix", default=" Answer yes or no.",
                    help="answer-format suffix (match the eval harness default)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-dir", default=None, help="default: runs/anchored_<model>_grpo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize-input", action="store_true",
                    help="LayerNorm the VIB input (auto-on for videollama2's massive activations)")
    args = ap.parse_args()
    args.save_dir = args.save_dir or f"runs/anchored_{args.model}_grpo"
    torch.manual_seed(args.seed)
    nin = args.normalize_input or args.model == "videollama2"

    m = get_model(args.model)
    bns, handles = attach_bottlenecks(m, cls=VariationalBottleneck, normalize_input=nin)
    opt = torch.optim.AdamW(bns.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    yes_id, no_id = letter_id(m, "yes"), letter_id(m, "no")
    abstain_id = letter_id(m, args.abstain_token) if args.abstain_token else None

    cats = ave.categories()
    items = ave.load_ave("train")
    rng = random.Random(args.seed)
    rng.shuffle(items)
    train_items = items[:args.pairs]
    probe = [_yesno_item(m, it, cats, random.Random(1), args.yn_suffix) for it in items[400:440]]
    print(f"train items: {len(train_items)} | probe: {len(probe)} | group={args.group} | "
          f"abstain={'on(' + args.abstain_token + ')' if abstain_id is not None else 'off'}", flush=True)

    set_bypass(bns, True)  # true frozen base
    bns.eval()
    base_yes, base_acc = _frac_yes(m, probe)
    bns.train()
    set_bypass(bns, False)
    print(f"[probe base] frac_yes={base_yes:.2f} acc={base_acc:.2f}  (collapse = frac_yes -> 0 or 1)",
          flush=True)

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train_items)
        for i in range(0, len(train_items) - args.accum + 1, args.accum):
            batch = []
            for it in train_items[i:i + args.accum]:
                ex = _yesno_item(m, it, cats, rng, args.yn_suffix)
                ex.update(yes_id=yes_id, no_id=no_id, abstain_id=abstain_id)
                batch.append(ex)
            mt = grpo_step(m, bns, opt, batch, group=args.group, beta_kl=args.beta_kl,
                           lam_ref=args.lam_ref, r_correct=args.r_correct,
                           r_abstain=args.r_abstain, r_halluc=args.r_halluc)
            step += 1
            print(f"epoch {epoch} step {step}: "
                  + "  ".join(f"{k}={v:+.4f}" for k, v in mt.items()), flush=True)
            if step % args.eval_every == 0:
                bns.eval()
                fy, ac = _frac_yes(m, probe)
                bns.train()
                ckpt = os.path.join(args.save_dir, f"bottleneck_step{step}.pt")
                torch.save({"state_dict": bns.state_dict(), "dim": m.hidden_dim,
                            "cls": "VariationalBottleneck", "model": args.model,
                            "normalize_input": nin}, ckpt)
                flag = "  <-- COLLAPSE ALARM" if (fy < 0.15 or fy > 0.85) else ""
                print(f"[probe step {step}] frac_yes={fy:.2f} (base {base_yes:.2f}) acc={ac:.2f}  "
                      f"saved {os.path.basename(ckpt)}{flag}", flush=True)

    for h in handles:
        h.remove()
    print(f"=== GRPO training done === eval: EXP=grpo bash scripts/eval_one.sh {args.model} <step>",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
