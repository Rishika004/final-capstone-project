#!/usr/bin/env bash
# Why does a near-perfectly surviving subspace not buy protection?
#  1. measure the ACTUAL survival (mel PSR) of every crafted perturbation
#  2. transfer of the same files to the other purifier (v2/DDPM, unseen)
#  3. lighter regulariser (lambda=1) at 20 speakers
#  4. the winning method + noise-matched regulariser at 20 speakers
cd "$(dirname "$0")/.."
V3=checkpoints/purifier_v3.pt
V2=checkpoints/purifier_v2.pt
TAGS=plain_m30_f15,null_m30_f15,nsT_m30_f15_l10,full_m30_f15,fullnull_m30_f15
say() { echo "[queue9] $(date +%H:%M:%S) $*"; }

say "1. survival + protection vs v3/ddim (no crafting)"
python experiments/eval_saved.py --tags $TAGS --purifier-ckpt $V3 --sampler ddim \
   --t 0,50,100,200 --thresholds 0.897,0.346,0.735 --out experiments/results/eval_psr_v3.csv \
   > experiments/results/eval_psr_v3.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/eval_psr_v3.log | sed -n '/perturbation survival/,$p'

say "2. transfer of the same files to purifier v2/ddpm (unseen by these crafters)"
python experiments/eval_saved.py --tags $TAGS --purifier-ckpt $V2 --sampler ddpm \
   --t 50,200 --thresholds 0.889,0.324,0.717 --out experiments/results/eval_transfer_v2.csv \
   > experiments/results/eval_transfer_v2.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/eval_transfer_v2.log | sed -n '/perturbation survival/,$p'

BASE="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 20 --steps 200 \
 --purifier-ckpt $V3 --sampler ddim --thresholds 0.897,0.346,0.735 --eval-t 50,100,200 \
 --thr-aware --weights 1,1,1 --out experiments/results/main_table_v3.csv"
run() {
  tag=$1; shift
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" > experiments/results/q9_${tag}.log 2>&1
  say "done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/q9_${tag}.log | grep "t= 50\|t=100\|t=200\|quality"
}
run nsT_m30_f15_l1     --variant plain --margin-db 30 --snr-floor 15 --lam-null 1  --null-t 200
run fullnsT_m30_f15_l10 --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
say "all done"
