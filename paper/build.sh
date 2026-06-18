#!/bin/bash
# Build the paper PDF from main.tex. Needs a TeX install with pdflatex + tikz/pgf.
#   bash paper/build.sh          ->  paper/main.pdf
#
# Get a TeX toolchain if you don't have one:
#   conda:   conda install -c conda-forge texlive-core      # works on the ABCI cluster
#   debian:  apt-get install texlive-latex-base texlive-latex-extra texlive-pictures \
#                            texlive-fonts-recommended
#   docker:  docker run --rm -v "$PWD":/w -w /w/paper texlive/texlive latexmk -pdf main.tex
#   none:    paste main.tex into https://overleaf.com
set -euo pipefail
cd "$(dirname "$0")"

# Prefer tectonic: self-contained, fetches packages on demand, no fmtutil/format-file
# pain (conda's texlive-core is notoriously broken here). Install: conda install -c
# conda-forge tectonic
if command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
  echo "-> $(pwd)/main.pdf"; exit 0
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "no tectonic and no pdflatex on PATH -- see the install hints at the top of this script." >&2
  echo "quickest: paste main.tex into https://overleaf.com" >&2
  exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex   # 2nd pass resolves refs
fi

echo "-> $(pwd)/main.pdf"
