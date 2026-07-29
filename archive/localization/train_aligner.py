#!/usr/bin/env python
"""Train the AV alignment head on FROZEN tokens -- now with NEGATIVES + a detection loss.

v1 (contrastive-only) made the map move with audio but FAILED the silence control: the
absolute alignment was reversed (det_silence > det_swap > det_match). Fix (SLAVC-style):
add silence + mismatched-event audio as explicit negatives and a margin detection loss
that drives the matched max-cosine ABOVE the swap/silence max-cosine, so the peak
similarity becomes a real "is this sound here?" confidence.

Pipeline: cache frozen tokens (+ one silence audio vector) once; train symmetric InfoNCE
+ a detection-margin loss; report corr(match,swap), peak, and det_{match,swap,silence}
BEFORE vs AFTER; save the aligner.

  python scripts/train_aligner.py --n 150 --epochs 40 --margin 0.1 --det-w 1.0
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


def _pool(t):
    return t.float().reshape(-1, t.shape[-1]).mean(0).half().cpu()


def _extract(model, items, cache_path):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if isinstance(d, dict) and "feats" in d and len(d["feats"]) >= 0.9 * len(items):
            print(f"loading {len(d['feats'])} cached features <- {cache_path}", flush=True)
            return d["feats"], d.get("silence")
        print("cache missing silence or too small for --n; re-extracting", flush=True)

    adapters = model.adapter_modules()
    cap: dict = {}

    def hook(name):
        def h(_m, _i, out):
            cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return h

    handles = [adapters["audio"].register_forward_hook(hook("audio")),
               adapters["vision"].register_forward_hook(hook("vision"))]
    lm = getattr(model.model, "thinker", model.model)

    def fwd(video_path):
        cap.clear()
        msg = model.message(video=video_path, prompt="What do you see and hear?")
        inputs = model.build_inputs(msg, use_audio_in_video=True)
        with torch.no_grad():
            lm(**inputs)
        return inputs

    feats = []
    for i, it in enumerate(items):
        inputs = fwd(it["video_path"])
        if "audio" not in cap or "vision" not in cap or inputs.get("video_grid_thw") is None:
            continue
        feats.append({"a": _pool(cap["audio"]),
                      "V": cap["vision"].float().reshape(-1, cap["vision"].shape[-1]).half().cpu(),
                      "grid": inputs["video_grid_thw"][0].tolist(),
                      "cat": it["category"], "path": it["video_path"]})
        if (i + 1) % 20 == 0:
            print(f"  extracted {len(feats)}/{i + 1}", flush=True)

    silence = None
    if feats:  # one silence audio vector: silence a real clip so the audio encoder still runs
        try:
            from rlvib.data.pairs import silence_audio
            sil_path = cache_path + ".silent.mp4"
            silence_audio(feats[0]["path"], sil_path)
            fwd(sil_path)
            if "audio" in cap:
                silence = _pool(cap["audio"])
                print("extracted silence audio vector", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"silence extraction failed: {e}", flush=True)

    for h in handles:
        h.remove()
    torch.save({"feats": feats, "silence": silence}, cache_path)
    print(f"cached {len(feats)} features (+silence={silence is not None}) -> {cache_path}", flush=True)
    return feats, silence


def _simmap_raw(a, V):
    a = a / (a.norm() + 1e-6)
    v = V / (V.norm(dim=-1, keepdim=True) + 1e-6)
    return v @ a


def _corr(x, y):
    x, y = x.reshape(-1).cpu().numpy(), y.reshape(-1).cpu().numpy()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _eval(simfn, feats, dev, silence=None):
    """mean corr(match,swap), peak, det_match, det_swap, det_silence over val clips."""
    rng = random.Random(1)
    by_cat: dict = {}
    for f in feats:
        by_cat.setdefault(f["cat"], []).append(f)
    cs, pk, dm, ds, dsil = [], [], [], [], []
    sil_t = silence.float().to(dev) if silence is not None else None
    for f in feats:
        other = [c for c in by_cat if c != f["cat"]]
        if not other:
            continue
        g = rng.choice(by_cat[rng.choice(other)])
        Vi = f["V"].float().to(dev)
        Mm, Ms = simfn(f["a"].float().to(dev), Vi), simfn(g["a"].float().to(dev), Vi)
        cs.append(_corr(Mm, Ms))
        pk.append(float((Mm.max() - Mm.mean()) / (Mm.std() + 1e-6)))
        dm.append(float(Mm.max()))
        ds.append(float(Ms.max()))
        if sil_t is not None:
            dsil.append(float(simfn(sil_t, Vi).max()))

    def M(xs):
        return float(np.nanmean(xs)) if xs else float("nan")

    return M(cs), M(pk), M(dm), M(ds), M(dsil)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.1, help="detection margin (cosine)")
    ap.add_argument("--det-w", type=float, default=1.0, help="detection-loss weight")
    ap.add_argument("--cache", default="runs/aligner_feats.pt")
    ap.add_argument("--save", default="runs/aligner.pt")
    args = ap.parse_args()

    m = get_model(args.model)
    dev = m.device
    items = ave.load_ave("train")
    rng = random.Random(0)
    rng.shuffle(items)
    feats, silence = _extract(m, items[: args.n], args.cache)
    rng.shuffle(feats)
    nval = max(8, len(feats) // 5)
    val, train = feats[:nval], feats[nval:]
    dim = feats[0]["a"].shape[-1]
    print(f"features: {len(feats)} (train {len(train)} / val {len(val)})  dim={dim}  "
          f"silence={silence is not None}", flush=True)

    aligner = AVAligner(dim=dim, proj=args.proj).to(dev).float()
    opt = torch.optim.AdamW(aligner.parameters(), lr=args.lr)

    bc, bp, bdm, bds, bdsil = _eval(_simmap_raw, val, dev, silence)
    print(f"[before] corr(match,swap)={bc:+.3f} peak={bp:+.2f} | "
          f"det_match={bdm:+.3f} det_swap={bds:+.3f} det_silence={bdsil:+.3f}", flush=True)

    sil_t = silence.float().to(dev) if silence is not None else None
    for ep in range(args.epochs):
        rng.shuffle(train)
        aligner.train()
        tot, acc, gap, nb = 0.0, 0.0, 0.0, 0
        for s in range(0, len(train) - args.batch + 1, args.batch):
            batch = train[s:s + args.batch]
            za = aligner.audio(torch.stack([b["a"].float().to(dev) for b in batch]))
            Vps = [aligner.visual(b["V"].float().to(dev)) for b in batch]
            B = len(batch)
            S = za.new_empty(B, B)
            for j, Vp in enumerate(Vps):
                S[:, j] = (za @ Vp.t()).max(dim=1).values  # S[i,j] = max-sim(audio_i, V_j)
            tgt = torch.arange(B, device=dev)
            infonce = 0.5 * (F.cross_entropy(S / args.temp, tgt)
                             + F.cross_entropy(S.t() / args.temp, tgt))

            det_match = S.diag()                       # max-sim(audio_i, V_i)
            S_off = S.clone()
            S_off.fill_diagonal_(-1e9)
            det_swap = S_off.max(dim=0).values         # per video i, hardest mismatched audio
            det_loss = F.relu(args.margin - (det_match - det_swap)).mean()
            if sil_t is not None:
                sa = aligner.audio(sil_t)
                det_sil = torch.stack([(Vp @ sa).max() for Vp in Vps])
                det_loss = det_loss + F.relu(args.margin - (det_match - det_sil)).mean()
                gap += float((det_match - det_sil).mean())

            loss = infonce + args.det_w * det_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            acc += float((S.argmax(1) == tgt).float().mean())
            nb += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"epoch {ep}: loss={tot/max(1,nb):.4f} top1={acc/max(1,nb):.3f} "
                  f"det_gap(match-silence)={gap/max(1,nb):+.3f}", flush=True)

    aligner.eval()
    with torch.no_grad():
        ac, apk, adm, ads, adsil = _eval(lambda a, V: aligner.simmap(a, V), val, dev, silence)
    print(f"[after ] corr(match,swap)={ac:+.3f} peak={apk:+.2f} | "
          f"det_match={adm:+.3f} det_swap={ads:+.3f} det_silence={adsil:+.3f}", flush=True)
    print(f"         want det_match > det_swap > det_silence  "
          f"(det_match-det_silence now {adm - adsil:+.3f}, was {bdm - bdsil:+.3f})", flush=True)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"state_dict": aligner.state_dict(), "dim": dim, "proj": args.proj}, args.save)
    print(f"saved aligner -> {args.save}", flush=True)
    print("=== aligner training done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
