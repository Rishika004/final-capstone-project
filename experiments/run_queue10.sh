#!/usr/bin/env bash
# Complete the 2-budget x method grid for the combined method (EOT + noise-matched
# null-space regulariser), which beat EOT alone at strong purification.
cd "$(dirname "$0")/.."
BASE="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 20 --steps 200 \
 --purifier-ckpt checkpoints/purifier_v3.pt --sampler ddim --thresholds 0.897,0.346,0.735 \
 --eval-t 50,100,200 --thr-aware --weights 1,1,1 --out experiments/results/main_table_v3.csv"
run() {
  tag=$1; shift
  echo "[queue10] $(date +%H:%M:%S) start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" > experiments/results/q10_${tag}.log 2>&1
  echo "[queue10] $(date +%H:%M:%S) done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/q10_${tag}.log | grep "t= 50\|t=100\|t=200\|quality"
}
run fullnsT_m20_f20_l10 --variant full --vocoder phase --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200
run fullnsT_m30_f15_l30 --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 30 --null-t 200
echo "[queue10] all done"
