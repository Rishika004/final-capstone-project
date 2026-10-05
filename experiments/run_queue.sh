#!/usr/bin/env bash
# Sequential GPU queue (ONE job at a time: the 6 GB laptop GPU thrashes with 2+).
cd "$(dirname "$0")/.."
COMMON="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 3 --steps 200 \
 --purifier-ckpt checkpoints/purifier_v2.pt --thresholds 0.889,0.324,0.717 --eval-t 50,100"
run() { # tag, then args
  tag=$1; shift
  echo "[queue] $(date +%H:%M:%S) start $tag"
  python experiments/exp_purifier_aware.py $COMMON --tag $tag "$@" > experiments/results/pa_${tag}_sweep.log 2>&1
  echo "[queue] $(date +%H:%M:%S) done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/pa_${tag}_sweep.log | grep "t= 50\|t=100\|quality"
}
run full_phase_v2  --variant full --vocoder phase
run full_f15_v2    --variant full --vocoder phase --snr-floor 15
run plain_m10_f20_v2 --variant plain --margin-db 10 --snr-floor 20
run plain_m20_f20_v2 --variant plain --margin-db 20 --snr-floor 20
run plain_m10_f25_v2 --variant plain --margin-db 10 --snr-floor 25
run plain_m20_f25_v2 --variant plain --margin-db 20 --snr-floor 25
echo "[queue] all done"
