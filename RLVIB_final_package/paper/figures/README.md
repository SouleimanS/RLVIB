# Figures used by `paper/final.tex`

`final.tex` degrades gracefully: `\figslot` and `\frm` draw a labelled placeholder box when a file
is absent, so the document **always compiles**. Missing files show as grey boxes rather than errors.

| File(s) | Used on | Status | How to (re)generate |
|---|---|---|---|
| `ll_shift.png` | Mechanism II — likelihood shift | **included** | `qsub scripts/probe_ll_shift.qsub` (regenerates from measured data) |
| `attn_av.png` | Mechanism I — attention to AV tokens | must be generated | `python scripts/attn_av_analysis.py --model qwen3-omni --modality audio` then `python scripts/plot_attn_av.py` |
| `avh_a2v_{1,2,3}.jpg` | Problem statement (example row) | must be extracted | `python scripts/find_avh_examples.py` — extracts three frames per illustrative AVHBench clip |
| `avh_v2a_{1,2,3}.jpg` | Problem statement (example row) | must be extracted | same |
| `as_flute_{1,2,3}.jpg` | Training data (example row) | must be extracted | `python scripts/swap_examples.py` — frames from an audio-substituted AVE clip |

The frame images are illustrative screenshots of publicly available benchmark clips and are not
redistributed with this package. Both extraction scripts write directly into this directory.

To confirm what the current build is missing:

```bash
cd paper && pdflatex -interaction=nonstopmode final.tex >/dev/null 2>&1
grep -c "figures/" final.log      # counts placeholder fallbacks that were drawn
```
