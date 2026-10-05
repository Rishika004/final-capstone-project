#!/usr/bin/env bash
# Replication of the protection test against the SECOND foreign purifier
# (philsyn SC09 DiffWave). One foreign purifier cannot carry a positive claim.
#
# NOTE the --tag: calibrate_thresholds.py names its output by CHANNEL, so without
# a tag this run would overwrite thresholds_diffwave_t*.csv from the LibriSpeech
# run -- and the "missing file" guard below would then pass on a STALE file from
# the wrong purifier, which is worse than crashing.
cd "$(dirname "$0")/.."
R=experiments/results
CK=checkpoints/foreign/diffwave_sc09_philsyn.pkl
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done

for T in 1 3; do
  rm -f $R/thresholds_diffwave_t${T}_sc09.csv
  echo "[pf-sc09] $(date +%H:%M:%S) calibrating thresholds, SC09 DiffWave t=$T"
  python experiments/calibrate_thresholds.py --channel diffwave --t-diffuse $T \
    --foreign-ckpt $CK --n-spk 20 --n-utt 3 --spk-offset 20 --tag sc09 \
    > $R/thr_sc09_t${T}.log 2>&1
  grep -viE "warn|weight_norm|detach" $R/thr_sc09_t${T}.log | tail -5
  if [ ! -f "$R/thresholds_diffwave_t${T}_sc09.csv" ]; then
    echo "[pf-sc09] FATAL: no thresholds for t=$T; aborting."; exit 1
  fi
done

for T in 1 3; do
  THR=$(python - <<PY
from utils.csvio import read_csv
d = read_csv("experiments/results/thresholds_diffwave_t${T}_sc09.csv")
m = {r.encoder: r.threshold for r in d.itertuples()}
print(",".join(str(m[k]) for k in ("WavLM-SV","ECAPA-TDNN","Resemblyzer d-vector")))
PY
)
  echo "[pf-sc09] $(date +%H:%M:%S) protection, SC09 t=$T, thresholds $THR"
  python experiments/eval_saved.py \
    --tags plain_m30_f15,nsT_m30_f15_l10,full_m30_f15,fullnsT_m30_f15_l10 \
    --t $T --seeds 2026,7 --n-files 20 --spk-offset 20 --utt-index 3 \
    --diffwave-ckpt $CK --thresholds "$THR" \
    --out $R/protect_foreign_sc09_t${T}.csv > $R/protect_sc09_t${T}.log 2>&1
done
echo "[pf-sc09] done"
