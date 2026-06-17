#!/usr/bin/env python
"""Train a small audio-visual ALIGNMENT head on FROZEN Qwen adapter tokens so that
cosine(audio, visual_patch) localizes the sounding source. Only f_a/f_v train.

Pipeline:
  1. extract & CACHE frozen tokens for N AVE clips (pooled audio a_i, visual patches V_i);
  2. train f_a/f_v CONTRASTIVELY -- matched audio<->video is the positive, the other
     clips in the batch are negatives, MIL-max over patches, symmetric InfoNCE;
  3. eval: contrastive top-1 acc + corr(match,swap) BEFORE (raw cosine) vs AFTER (trained)
     + map peakiness; save the aligner.

  python scripts/train_aligner.py --n 150 --epochs 30
  # then visualize the trained maps with the same swap control:
  python scripts/localize_cosine.py --aligner runs/aligner.pt
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from rlvib.data import ave
from rlvib.models import get_model
from rlvib.models.aligner import AVAligner


def _extract(model, items, cache_path):
    if os.path.exists(cache_path):
        print(f"loading cached features <- {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)
    adapters = model.adapter_modules()
    cap: dict = {}

    def hook(name):
        def h(_m, _i, out):
            cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return h

    handles = [adapters["audio"].register_forward_hook(hook("audio")),
               adapters["vision"].register_forward_hook(hook("vision"))]
    lm = getattr(model.model, "thinker", model.model)
    feats = []
    for i, it in enumerate(items):
        cap.clear()
        msg = model.message(video=it["video_path"], prompt="What do you see and hear?")
        inputs = model.build_inputs(msg, use_audio_in_video=True)
        with torch.no_grad():
            lm(**inputs)
        if "audio" not in cap or "vision" not in cap or inputs.get("video_grid_thw") is None:
            continue
        A = cap["audio"].float().reshape(-1, cap["audio"].shape[-1])
        V = cap["vision"].float().reshape(-1, cap["vision"].shape[-1])
        feats.append({"a": A.mean(0).half().cpu(), "V": V.half().cpu(),
                      "grid": inputs["video_grid_thw"][0].tolist(),
                      "cat": it["category"], "path": it["video_path"]})
        if (i + 1) % 20 == 0:
            print(f"  extracted {len(feats)}/{i + 1}", flush=True)
    for h in handles:
        h.remove()
    torch.save(feats, cache_path)
    print(f"cached {len(feats)} features -> {cache_path}", flush=True)
    return feats


def _simmap_raw(a, V):
    a = a / (a.norm() + 1e-6)
    v = V / (V.norm(dim=-1, keepdim=True) + 1e-6)
    return v @ a


def _corr(x, y):
    x, y = x.reshape(-1).cpu().numpy(), y.reshape(-1).cpu().numpy()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _swap_eval(simfn, feats, dev):
    """mean corr(M_match, M_swap) + mean peak over val clips (fixed swap partners)."""
    rng = random.Random(1)
    by_cat: dict = {}
    for f in feats:
        by_cat.setdefault(f["cat"], []).append(f)
    cs, pk = [], []
    for f in feats:
        other = [c for c in by_cat if c != f["cat"]]
        if not other:
            continue
        g = rng.choice(by_cat[rng.choice(other)])
        Vi = f["V"].float().to(dev)
        Mm = simfn(f["a"].float().to(dev), Vi)
        Ms = simfn(g["a"].float().to(dev), Vi)
        cs.append(_corr(Mm, Ms))
        pk.append(float((Mm.max() - Mm.mean()) / (Mm.std() + 1e-6)))
    return float(np.nanmean(cs)) if cs else float("nan"), float(np.nanmean(pk)) if pk else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--cache", default="runs/aligner_feats.pt")
    ap.add_argument("--save", default="runs/aligner.pt")
    args = ap.parse_args()

    m = get_model(args.model)
    dev = m.device
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)
    feats = _extract(m, items[: args.n], args.cache)
    rng.shuffle(feats)
    nval = max(8, len(feats) // 5)
    val, train = feats[:nval], feats[nval:]
    dim = feats[0]["a"].shape[-1]
    print(f"features: {len(feats)} (train {len(train)} / val {len(val)})  dim={dim}", flush=True)

    aligner = AVAligner(dim=dim, proj=args.proj).to(dev).float()
    opt = torch.optim.AdamW(aligner.parameters(), lr=args.lr)

    b_corr, b_peak = _swap_eval(_simmap_raw, val, dev)
    print(f"[before] corr(match,swap)={b_corr:+.3f}  peak={b_peak:+.2f}", flush=True)

    for ep in range(args.epochs):
        rng.shuffle(train)
        aligner.train()
        tot, acc, nb = 0.0, 0.0, 0
        for s in range(0, len(train) - args.batch + 1, args.batch):
            batch = train[s:s + args.batch]
            za = aligner.audio(torch.stack([b["a"].float().to(dev) for b in batch]))  # (B,proj)
            Vps = [aligner.visual(b["V"].float().to(dev)) for b in batch]
            B = len(batch)
            S = za.new_empty(B, B)
            for j, Vp in enumerate(Vps):
                S[:, j] = (za @ Vp.t()).max(dim=1).values  # audio_i vs best patch of V_j
            S = S / args.temp
            tgt = torch.arange(B, device=dev)
            loss = 0.5 * (F.cross_entropy(S, tgt) + F.cross_entropy(S.t(), tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            acc += float((S.argmax(1) == tgt).float().mean())
            nb += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"epoch {ep}: loss={tot / max(1, nb):.4f}  train_top1={acc / max(1, nb):.3f}",
                  flush=True)

    aligner.eval()
    with torch.no_grad():
        a_corr, a_peak = _swap_eval(lambda a, V: aligner.simmap(a, V), val, dev)
    print(f"[after ] corr(match,swap)={a_corr:+.3f}  peak={a_peak:+.2f}  "
          f"(want corr DOWN from {b_corr:+.3f}, peak UP from {b_peak:+.2f})", flush=True)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"state_dict": aligner.state_dict(), "dim": dim, "proj": args.proj}, args.save)
    print(f"saved aligner -> {args.save}", flush=True)
    print("=== aligner training done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
