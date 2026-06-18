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

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "no pdflatex on PATH -- see the install hints at the top of this script." >&2
  exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex   # 2nd pass resolves refs
fi

echo "-> $(pwd)/main.pdf"
