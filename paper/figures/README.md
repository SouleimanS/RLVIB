# Figures used by `paper/final.tex`

`final.tex` degrades gracefully: `\figslot` and `\frm` draw a labelled placeholder box when a file
is absent, so the document **always compiles**. Missing files show as grey boxes rather than errors.

| File(s) | Used on | Status | How to regenerate |
|---|---|---|---|
| `ll_shift.png` | Mechanism II — likelihood shift | **included** | `qsub scripts/probe_ll_shift.qsub` (regenerates from measured data) |
| `attn_av.png` | Mechanism I — attention to AV tokens | **included** | `python scripts/attn_av_analysis.py --model qwen3-omni --modality audio` then `python scripts/plot_attn_av.py` |
| `avh_a2v_{1,2,3}.jpg` | Problem statement (example row) | **included** | `python scripts/find_avh_examples.py` — extracts three frames per illustrative AVHBench clip |
| `avh_v2a_{1,2,3}.jpg` | Problem statement (example row) | **included** | same |
| `as_flute_{1,2,3}.jpg` | Training data (example row) | **included** | `python scripts/swap_examples.py` — frames from an audio-substituted AVE clip |

All ten files are extracted stills/plots produced on the cluster (dataset + GPU + trained
checkpoint required), then copied here — `find_avh_examples.py` and `swap_examples.py` need
`data/AVE/` and `runs/*.json`/a trained bottleneck checkpoint that only exist there. If any file
is later regenerated, replace it here.

To confirm what the current build is missing:

```bash
cd paper && pdflatex -interaction=nonstopmode final.tex >/dev/null 2>&1
grep -c "figures/" final.log      # counts placeholder fallbacks that were drawn
```

## Brand assets used by `paper/report.tex` (École Polytechnique template)

`report.tex` uses the official `polytechnique.sty` document class (copied in at
`paper/polytechnique.sty`), which requires five image files under `pdflatex`. Four are the real
official files, supplied directly by the user:

| File | Role | Status |
|---|---|---|
| `polytechnique-logovert.pdf` | vertical logo, title-page bottom | **real** |
| `polytechnique-logohori.pdf` | horizontal logo, running header on every page | **real** |
| `polytechnique-filetlongrouge.pdf` | red decorative rule under `\section` headings | **real** |
| `polytechnique-filetcourt.pdf` | short blue rule under `\subsection` headings and on the title page | **real** |
| `polytechnique-armes.pdf` | large pale coat-of-arms watermark on the title page | **placeholder** — see below |

`polytechnique-armes.pdf` is **not** the official coat of arms; it was never supplied. It's a
locally-generated stand-in (light-grey dashed shield outline with "MISSING / replace with the
official coat of arms file" text) so the document compiles cleanly instead of erroring or showing
a broken-image box. Swap in the real file under this exact name and rebuild — no other change is
needed.

`typographix.pdf` (the LaTeX users'-group logo, from the same upload) is present here but **not**
currently wired into `report.tex` via `\logo{}` — it's unrelated to the report's actual host
institution (AIST CVRT) and was left out rather than guessed in. Delete it or wire it in via
`\logo{typographix}` in the preamble if a second header logo is wanted.
