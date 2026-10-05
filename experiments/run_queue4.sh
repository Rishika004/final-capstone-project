#!/usr/bin/env bash
# Threshold-aware ensemble loss: 3-speaker checks, then the 20-speaker main table.
cd "$(dirname "$0")/.."
BASE="--dataset librispeech --spk-offset 20 --utt-index 3 --steps 200 \
 --purifier-ckpt checkpoints/purifier_v2.pt --thresholds 0.889,0.324,0.717 --eval-t 50,100 \
 --thr-aware --weights 1,1,1"
run() { # out-csv  n-files  tag  args...
  out=$1; n=$2; tag=$3; shift 3
  echo "[queue4] $(date +%H:%M:%S) start $tag (n=$n)"
  python experiments/exp_purifier_aware.py $BASE --out $out --n-files $n --tag $tag "$@" \
      > experiments/results/q4_${tag}.log 2>&1
  echo "[queue4] $(date +%H:%M:%S) done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/q4_${tag}.log | grep "t= 50\|t=100\|quality"
}
T=experiments/results/purifier_aware.csv
M=experiments/results/main_table.csv
run $T 3  plain_m20_f20_ta --variant plain --margin-db 20 --snr-floor 20
run $T 3  full_m20_f20_ta  --variant full --vocoder phase --margin-db 20 --snr-floor 20
run $M 20 plain_m20_f20    --variant plain --margin-db 20 --snr-floor 20
run $M 20 full_m20_f20     --variant full --vocoder phase --margin-db 20 --snr-floor 20
run $M 20 plain_m30_f15    --variant plain --margin-db 30 --snr-floor 15
run $M 20 full_m30_f15     --variant full --vocoder phase --margin-db 30 --snr-floor 15
echo "[queue4] all done"
