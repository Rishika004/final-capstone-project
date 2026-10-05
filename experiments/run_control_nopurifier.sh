#!/usr/bin/env bash
# CONTROL for the "constraint helps against a foreign purifier" finding.
#
# Alternative explanation to kill: maybe the pass-through-constrained perturbation
# is simply MORE ROBUST IN GENERAL -- to any channel -- rather than specifically
# insulated against purifier mismatch. If so it would also beat plain EOT with NO
# purifier in the loop at all.
#
# --diffwave-ckpt with --t 0 makes the purifier the identity, so this measures the
# published waveform straight into the encoders: no purification, no vocoder.
# Thresholds must therefore be the CLEAN-channel ones.
cd "$(dirname "$0")/.."
R=experiments/results
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done
rm -f $R/thresholds_clean.csv
echo "[ctl] $(date +%H:%M:%S) calibrating clean-channel thresholds"
python experiments/calibrate_thresholds.py --channel clean --n-spk 20 --n-utt 3 \
  --spk-offset 20 > $R/thr_clean.log 2>&1
grep -viE "warn|weight_norm|detach" $R/thr_clean.log | tail -5
[ -f $R/thresholds_clean.csv ] || { echo "[ctl] FATAL: no clean thresholds"; exit 1; }
THR=$(python - <<PY
from utils.csvio import read_csv
d = read_csv("experiments/results/thresholds_clean.csv")
m = {r.encoder: r.threshold for r in d.itertuples()}
print(",".join(str(m[k]) for k in ("WavLM-SV","ECAPA-TDNN","Resemblyzer d-vector")))
PY
)
echo "[ctl] $(date +%H:%M:%S) no-purifier control, thresholds $THR"
python experiments/eval_saved.py \
  --tags plain_m30_f15,nsT_m30_f15_l10,full_m30_f15,fullnsT_m30_f15_l10 \
  --t 0 --seeds 2026 --n-files 20 --spk-offset 20 --utt-index 3 \
  --diffwave-ckpt checkpoints/foreign/diffwave_libri_deantifake.pkl \
  --thresholds "$THR" --out $R/control_nopurifier.csv > $R/control_nopurifier.log 2>&1
echo "[ctl] done"
