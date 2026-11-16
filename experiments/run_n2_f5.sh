#!/usr/bin/env bash
# N2 -- the modern-cloner column. PREREG: paper/PREREG_N2_modern_cloner.md
#
# Three interpreters on purpose:
#   stage A, C, D  project Python   (purifiers, encoders, thresholds, stats)
#   stage B        external/f5tts_venv  (F5-TTS and its own torch pins)
#
# One GPU job at a time -- the stages are sequential and each waits.
set -u
cd "$(dirname "$0")/.." || exit 1
R=experiments/results
PREREG=paper/PREREG_N2_modern_cloner.md
F5PY=external/f5tts_venv/Scripts/python.exe
say() { echo "[$(date +%H:%M:%S)] $*"; }

export TRANSFORMERS_NO_TF=1

if [ ! -x "$F5PY" ]; then
  echo "FAIL: $F5PY not found. Run external/install_f5tts.sh first." >&2
  exit 1
fi

say "stage A -- prompts + purified references"
if [ -f $R/f5_prompts.csv ]; then
  say "  skip (f5_prompts.csv present)"
else
  python experiments/prep_f5_prompts.py --prereg $PREREG \
    > $R/f5_prep.log 2>&1 || { say "stage A FAILED (see $R/f5_prep.log)"; exit 1; }
fi
say "  $(( $(wc -l < $R/f5_prompts.csv) - 1 )) prompt rows"

say "stage B -- F5-TTS generation (isolated venv)"
"$F5PY" experiments/f5_clone.py --prompts $R/f5_prompts.csv \
  > $R/f5_clone.log 2>&1
rc=$?
say "  exit $rc; $(ls $R/f5_clones 2>/dev/null | wc -l) clones on disk"
[ $rc -ne 0 ] && { say "stage B had failures -- see $R/f5_clone.log"; }

say "stage C -- score with the project's encoders"
python experiments/score_f5_clones.py --prereg $PREREG \
  > $R/f5_score.log 2>&1 || { say "stage C FAILED (see $R/f5_score.log)"; exit 1; }
tail -30 $R/f5_score.log

say "stage D -- the pre-registered analysis"
python experiments/analyse_f5.py | tee $R/f5_analysis.txt

say "N2 DONE. Read the GATE first; it decides whether any verdict may be read."
