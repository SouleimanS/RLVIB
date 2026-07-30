#!/usr/bin/env bash
# Self-check for the RLVIB reproduction package. Run from the package root.
# Confirms the code is complete and importable; does NOT need a GPU or the datasets.
set -u
cd "$(dirname "$0")"
fail=0
say() { printf "%-52s %s\n" "$1" "$2"; }

echo "=== structure ==="
for p in README.md src/rlvib scripts baselines tests paper/final.tex \
         environment.yml environment-minicpm.yml environment-vllama2.yml pyproject.toml; do
  [ -e "$p" ] && say "$p" "OK" || { say "$p" "MISSING"; fail=1; }
done

echo; echo "=== python syntax (all sources parse) ==="
n=0; bad=0
while IFS= read -r f; do
  n=$((n+1)); python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null || { echo "  PARSE FAIL: $f"; bad=$((bad+1)); }
done < <(find src scripts baselines tests -name '*.py')
say "parsed $n python files" "$([ $bad -eq 0 ] && echo OK || echo "$bad FAILED")"
[ $bad -eq 0 ] || fail=1

echo; echo "=== shell/PBS scripts parse ==="
m=0; sbad=0
while IFS= read -r f; do
  m=$((m+1)); bash -n "$f" 2>/dev/null || { echo "  PARSE FAIL: $f"; sbad=$((sbad+1)); }
done < <(find scripts baselines -name '*.sh' -o -name '*.qsub')
say "parsed $m shell scripts" "$([ $sbad -eq 0 ] && echo OK || echo "$sbad FAILED")"
[ $sbad -eq 0 ] || fail=1

echo; echo "=== no references to excluded/archived code ==="
if grep -rn "models\.aligner\|rlvib\.data\.dave\|rlvib\.eval\.run_dave" src scripts baselines 2>/dev/null | grep -v '^\s*#' | grep -q .; then
  echo "  found dangling references"; fail=1
else say "no dangling imports" "OK"; fi

echo; echo "=== LaTeX structure ==="
python3 - <<'PY'
import re,sys
s=open("paper/final.tex").read()
ok = s.count(r"\begin{frame}")==s.count(r"\end{frame}") and s.count("{")==s.count("}")
print(f"{'final.tex: %d frames, balanced'%s.count(chr(92)+'begin{frame}'):52s} {'OK' if ok else 'BROKEN'}")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail=1

echo; echo "=== imports (needs the conda env; skipped if torch absent) ==="
PYTHONPATH=src python3 - <<'PY'
import importlib
mods=["rlvib.data.ave","rlvib.data.pairs","rlvib.data.mmau","rlvib.data.videomme",
      "rlvib.data.avshift","rlvib.eval.metrics","rlvib.models"]
bad=[]
for m in mods:
    try: importlib.import_module(m)
    except ModuleNotFoundError as e:
        if "torch" in str(e) or "pandas" in str(e): continue
        bad.append((m,str(e)[:60]))
    except Exception as e: bad.append((m,str(e)[:60]))
print(f"{'core modules importable':52s} {'OK' if not bad else 'FAILED'}")
for b in bad: print("   ",b)
PY

echo
[ $fail -eq 0 ] && echo "PACKAGE OK" || echo "PACKAGE HAS PROBLEMS (see above)"
exit $fail
