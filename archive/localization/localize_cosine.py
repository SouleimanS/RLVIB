#!/usr/bin/env python
"""Free audio->video localization via cosine similarity of FROZEN adapter tokens,
with the audio-swap faithfulness control. No training (parameter-free).

Mechanism (the canonical SSL map, applied to Qwen's adapter tokens):
  audio tokens A (T_a, d) from audio_tower.proj2 ; visual patch tokens V (T_v, d) from
  visual.merger, reshaped to (t, h, w). map = cos( mean(A), V ) per patch -> heatmap.

Faithfulness control (the decisive test): a clip's visual tokens are audio-independent,
so we keep V from clip i and swap only the pooled audio vector --
  M_match = cos(mean(A_i), V_i)     # the clip's own (matched) audio
  M_swap  = cos(mean(A_j), V_i)     # a DIFFERENT-event clip's audio, SAME frames
A genuine audio localizer => M_match localizes the source while M_swap moves/collapses,
so corr(M_match, M_swap) is LOW. Visual saliency => the map ignores audio => corr HIGH.
Also reports corr(M_match, ||V|| visual-saliency) and the peakiness of M_match.

  python scripts/localize_cosine.py --n 6
"""
from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
import torch

from rlvib.data import ave
from rlvib.models import get_model


def _san(s: str) -> str:
    """Filesystem-safe slug for an event name."""
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")[:28]


def _load_frames(video_path, t):
    import decord

    vr = decord.VideoReader(video_path)
    n = len(vr)
    idx = np.linspace(0, n - 1, t).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()


def _flat(x):
    x = x.float()
    return x.reshape(-1, x.shape[-1])  # (T, d)


def _cosmap(audio, visual, aligner=None):
    """cos(mean audio vector, each visual patch vector) -> (T_v,) in [-1, 1].
    With a trained aligner, project both through f_a/f_v first."""
    if aligner is not None:
        with torch.no_grad():
            a = aligner.audio(audio.mean(0).float())
            v = aligner.visual(visual.float())
        return (v @ a).cpu().numpy()
    a = audio.mean(0)
    a = a / (a.norm() + 1e-6)
    v = visual / (visual.norm(dim=-1, keepdim=True) + 1e-6)
    return (v @ a).cpu().numpy()


def _corr(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _render(plt, path, video_path, rows, title):
    """rows: list of (map (t,h,w), tag). One row per map, overlaid on sampled frames."""
    nrows = len(rows)
    t = rows[0][0].shape[0]
    nsel = min(t, 4)
    sel = np.linspace(0, t - 1, nsel).round().astype(int)
    try:
        frames = _load_frames(video_path, t)
    except Exception:  # noqa: BLE001
        frames = None

    def norm(a):
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    fig = plt.figure(figsize=(3.2 * nsel, 0.8 + 3.0 * nrows))
    fig.suptitle(title, fontsize=10)
    for r, (M, tag) in enumerate(rows):
        for col, fi in enumerate(sel):
            ax = fig.add_subplot(nrows, nsel, r * nsel + col + 1)
            heat = norm(M[fi])
            if frames is not None:
                H, W = frames[fi].shape[:2]
                ax.imshow(frames[fi], extent=(0, W, H, 0))
                ax.imshow(heat, cmap="turbo", alpha=0.5, extent=(0, W, H, 0),
                          interpolation="bilinear", vmin=0.0, vmax=1.0)
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)
            else:
                ax.imshow(heat, cmap="turbo", vmin=0.0, vmax=1.0)
            ax.set_title(f"{tag}  frame {fi}", fontsize=8)
            ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="runs/localize")
    ap.add_argument("--aligner", default=None, help="path to a trained AVAligner checkpoint")
    ap.add_argument("--silence", action="store_true",
                    help="also compute the silence control (peak should collapse with no audio)")
    args = ap.parse_args()

    m = get_model(args.model)
    os.makedirs(args.out, exist_ok=True)

    aligner = None
    if args.aligner:
        if not os.path.exists(args.aligner):
            raise SystemExit(f"aligner checkpoint not found: {args.aligner}\n"
                             f"Train it first:  qsub scripts/train_aligner.qsub  "
                             f"(writes runs/aligner.pt), then re-run this.")
        from rlvib.models.aligner import AVAligner
        ck = torch.load(args.aligner, weights_only=False)
        aligner = AVAligner(dim=ck["dim"], proj=ck["proj"]).to(m.device).float()
        aligner.load_state_dict(ck["state_dict"])
        aligner.eval()
        print(f"loaded aligner <- {args.aligner}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        plt = None
        print("matplotlib unavailable -> numbers only")

    vis = m.model.thinker.visual
    merge = (getattr(getattr(vis, "config", vis), "spatial_merge_size", None)
             or getattr(vis, "spatial_merge_size", 2))
    adapters = m.adapter_modules()
    cap: dict = {}

    def hook(name):
        def h(_mod, _inp, out):
            cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return h

    handles = [adapters["audio"].register_forward_hook(hook("audio")),
               adapters["vision"].register_forward_hook(hook("vision"))]
    lm = getattr(m.model, "thinker", m.model)

    def run(video_path):
        cap.clear()
        msg = m.message(video=video_path, prompt="What do you see and hear?")
        inputs = m.build_inputs(msg, use_audio_in_video=True)
        with torch.no_grad():
            lm(**inputs)
        if "audio" not in cap or "vision" not in cap:
            return None
        grid = inputs.get("video_grid_thw")
        return _flat(cap["audio"]), _flat(cap["vision"]), (None if grid is None else grid[0].tolist())

    items = ave.load_ave("train")
    rng = random.Random(7)
    rng.shuffle(items)
    by_cat: dict = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    corrs_swap, corrs_sal, peaks, corrs_sil, peaks_sil = [], [], [], [], []
    dets_m, dets_sw, dets_sil = [], [], []
    for k in range(min(args.n, len(items))):
        it = items[k]
        other = [c for c in by_cat if c != it["category"]]
        jt = rng.choice(by_cat[rng.choice(other)])
        ri = run(it["video_path"])
        rj = run(jt["video_path"])
        if ri is None or rj is None:
            print(f"[{k}] no audio/visual tokens, skip", flush=True)
            continue
        A_i, V_i, grid = ri
        A_j, _, _ = rj
        if grid is None:
            print(f"[{k}] no grid, skip", flush=True)
            continue
        t, h, w = grid
        hm, wm = h // merge, w // merge
        if t * hm * wm != V_i.shape[0]:
            print(f"[{k}] grid mismatch ({t*hm*wm} != {V_i.shape[0]}), skip", flush=True)
            continue
        shutil.copy(it["video_path"],  # the clips used, copied in so they can be inspected
                    os.path.join(args.out, f"clip{k}_match_{_san(it['category'])}.mp4"))
        shutil.copy(jt["video_path"],
                    os.path.join(args.out, f"clip{k}_swapaudio_{_san(jt['category'])}.mp4"))
        try:  # the actual mismatch to watch: clip i's video + clip j's audio
            from rlvib.data.pairs import swap_audio
            swap_audio(it["video_path"], jt["video_path"], os.path.join(
                args.out, f"clip{k}_swapped_seen-{_san(it['category'])}_heard-{_san(jt['category'])}.mp4"))
        except Exception as e:  # noqa: BLE001
            print(f"     swapped mux failed: {e}", flush=True)
        Mm = _cosmap(A_i, V_i, aligner).reshape(t, hm, wm)
        Ms = _cosmap(A_j, V_i, aligner).reshape(t, hm, wm)
        sal = V_i.norm(dim=-1).cpu().numpy().reshape(t, hm, wm)
        c_swap, c_sal = _corr(Mm, Ms), _corr(Mm, sal)
        peak = float((Mm.max() - Mm.mean()) / (Mm.std() + 1e-6))
        det_m, det_sw = float(Mm.max()), float(Ms.max())  # absolute detection score (max cosine)
        corrs_swap.append(c_swap)
        corrs_sal.append(c_sal)
        peaks.append(peak)
        dets_m.append(det_m)
        dets_sw.append(det_sw)
        rows = [(Mm, "match"), (Ms, "swap")]
        line = (f"[{k}] match={it['category'][:20]:20s} swap={jt['category'][:20]:20s} "
                f"corr(swap)={c_swap:+.3f} peak={peak:+.2f} det_match={det_m:+.3f} det_swap={det_sw:+.3f}")

        c_sil = peak_sil = float("nan")
        if args.silence:
            from rlvib.data.pairs import silence_audio
            sil_path = os.path.join(args.out, f"clip{k}_silent.mp4")
            rs = None
            try:
                silence_audio(it["video_path"], sil_path)
                rs = run(sil_path)
            except Exception as e:  # noqa: BLE001
                print(f"     silence build/run failed: {e}", flush=True)
            if rs is not None:
                Msil = _cosmap(rs[0], V_i, aligner).reshape(t, hm, wm)
                c_sil = _corr(Mm, Msil)
                peak_sil = float((Msil.max() - Msil.mean()) / (Msil.std() + 1e-6))
                det_sil = float(Msil.max())
                corrs_sil.append(c_sil)
                peaks_sil.append(peak_sil)
                dets_sil.append(det_sil)
                rows.append((Msil, "silence"))
                line += (f" | corr(silence)={c_sil:+.3f} peak_silence={peak_sil:+.2f} "
                         f"det_silence={det_sil:+.3f}")
        print(line, flush=True)

        if plt is not None:
            title = (f"match={it['category']}  |  swap={jt['category']}\n"
                     f"corr(match,swap)={c_swap:+.2f}  corr(match,sal)={c_sal:+.2f}  peak={peak:+.2f}"
                     + (f"   corr(match,silence)={c_sil:+.2f}  peak_silence={peak_sil:+.2f}"
                        if args.silence else "")
                     + "   (low corr + collapsing peak = audio-dependent)")
            _render(plt, os.path.join(args.out, f"clip{k}_localize.png"),
                    it["video_path"], rows, title)

    for hnd in handles:
        hnd.remove()

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nN={len(peaks)}  mean corr(match,swap)={_mean(corrs_swap):+.3f}  "
          f"mean corr(match,sal)={_mean(corrs_sal):+.3f}  mean peak={_mean(peaks):+.2f}")
    print(f"        detection (abs max cosine): det_match={_mean(dets_m):+.3f}  "
          f"det_swap={_mean(dets_sw):+.3f}"
          + (f"  det_silence={_mean(dets_sil):+.3f}" if dets_sil else ""))
    if corrs_sil:
        print(f"        silence: mean corr(match,silence)={_mean(corrs_sil):+.3f}  "
              f"peak_match={_mean(peaks):+.2f}  peak_silence={_mean(peaks_sil):+.2f}")
        print("        faithful => det_silence << det_match (alignment drops), not just peak shape")
    print("read: LOW corr(match,swap) => map CHANGES with audio => audio-dependent (good); "
          "HIGH => audio-invariant saliency.")
    print("      LOW corr(match,sal) => not just visual-norm saliency; HIGH peak => map is localized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
