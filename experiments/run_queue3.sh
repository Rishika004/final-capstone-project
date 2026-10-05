#!/usr/bin/env bash
# Main paper-table runs: 20 held-out LibriSpeech dev-clean speakers (ids sorted,
# index 20..39; never seen by purifier v2), one 4-s utterance each (utt-index 3,
# disjoint from the calibration utterances 0..2). Starts after queue2.
cd "$(dirname "$0")/.."
while ! grep -q "all done" experiments/results/queue2.log 2>/dev/null; do sleep 20; done
COMMON="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 20 --steps 200 \
 --purifier-ckpt checkpoints/purifier_v2.pt --thresholds 0.889,0.324,0.717 --eval-t 50,100 \
 --out experiments/results/main_table.csv"
run() {
  tag=$1; shift
  echo "[queue3] $(date +%H:%M:%S) start $tag"
  python experiments/exp_purifier_aware.py $COMMON --tag $tag "$@" > experiments/results/main_${tag}.log 2>&1
  echo "[queue3] $(date +%H:%M:%S) done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/main_${tag}.log | grep "t= 50\|t=100\|quality"
}
run plain_m20_f20 --variant plain --margin-db 20 --snr-floor 20
run full_m20_f20  --variant full --vocoder phase --margin-db 20 --snr-floor 20
run plain_m30_f15 --variant plain --margin-db 30 --snr-floor 15
run full_m30_f15  --variant full --vocoder phase --margin-db 30 --snr-floor 15
echo "[queue3] all done"
