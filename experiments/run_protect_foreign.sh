#!/usr/bin/env bash
# Protection measured through a FOREIGN purifier (De-AntiFake's DiffWave).
# Every protection number in the paper so far is against our own v3 surrogate.
# This asks the same question against a published purifier that operates on raw
# waveform -- a strictly stronger attacker, since it never pays the Griffin-Lim
# cost (E6: alpha 0.57-0.68).
#
# Step 1 RECALIBRATES EER thresholds through the DiffWave channel. This is not
# optional: thresholds calibrated through "our v3 + GL" are meaningless against a
# different channel, and reusing them would invalidate every number downstream.
cd "$(dirname "$0")/.."
R=experiments/results
CK=checkpoints/foreign/diffwave_libri_deantifake.pkl
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done

for T in 1 3; do
  echo "[pf] $(date +%H:%M:%S) calibrating thresholds through DiffWave t=$T"
  python experiments/calibrate_thresholds.py --channel diffwave --t-diffuse $T \
    --foreign-ckpt $CK --n-spk 20 --n-utt 3 --spk-offset 20 \
    > $R/thr_diffwave_t${T}.log 2>&1
  grep -viE "warn|weight_norm|detach" $R/thr_diffwave_t${T}.log | tail -6
  # Fail loudly. Continuing without thresholds would evaluate protection against
  # an empty threshold set and emit numbers that look like data. The first run of
  # this script did exactly that when --channel diffwave was missing from the
  # argparse choices list.
  if [ ! -f "$R/thresholds_diffwave_t${T}.csv" ]; then
    echo "[pf] FATAL: calibration produced no thresholds for t=$T; aborting."
    exit 1
  fi
done

for T in 1 3; do
  THR=$(python - <<PY
from utils.csvio import read_csv
d = read_csv("experiments/results/thresholds_diffwave_t${T}.csv")
m = {r.encoder: r.threshold for r in d.itertuples()}
print(",".join(str(m[k]) for k in ("WavLM-SV","ECAPA-TDNN","Resemblyzer d-vector")))
PY
)
  echo "[pf] $(date +%H:%M:%S) protection through DiffWave t=$T, thresholds $THR"
  python experiments/eval_saved.py \
    --tags plain_m30_f15,nsT_m30_f15_l10,full_m30_f15,fullnsT_m30_f15_l10 \
    --t $T --seeds 2026,7 --n-files 20 --spk-offset 20 --utt-index 3 \
    --diffwave-ckpt $CK --thresholds "$THR" \
    --out $R/protect_foreign_t${T}.csv > $R/protect_foreign_t${T}.log 2>&1
  grep -viE "warn|weight_norm|detach" $R/protect_foreign_t${T}.log | tail -20
done
echo "[pf] done"
