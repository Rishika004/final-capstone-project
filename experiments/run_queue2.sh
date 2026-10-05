#!/usr/bin/env bash
# Second sequential batch: purifier-aware at the operating points where plain
# reached PESQ >= 3, to test "more protection at equal perceptual budget".
cd "$(dirname "$0")/.."
while ! grep -q "all done" experiments/results/queue.log 2>/dev/null; do sleep 15; done
COMMON="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 3 --steps 200 \
 --purifier-ckpt checkpoints/purifier_v2.pt --thresholds 0.889,0.324,0.717 --eval-t 50,100"
run() {
  tag=$1; shift
  echo "[queue2] $(date +%H:%M:%S) start $tag"
  python experiments/exp_purifier_aware.py $COMMON --tag $tag "$@" > experiments/results/pa_${tag}_sweep.log 2>&1
  echo "[queue2] $(date +%H:%M:%S) done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/pa_${tag}_sweep.log | grep "t= 50\|t=100\|quality"
}
run full_m20_f20_v2  --variant full --vocoder phase --margin-db 20 --snr-floor 20
run plain_m15_f20_v2 --variant plain --margin-db 15 --snr-floor 20
run full_m15_f20_v2  --variant full --vocoder phase --margin-db 15 --snr-floor 20
echo "[queue2] all done"
