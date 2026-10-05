#!/usr/bin/env bash
# Re-run E4 with the signal-gain metric so the paper's survival table and the
# double-dissociation table use ONE metric (RESULTS section 11a).
cd "$(dirname "$0")/.."
R=experiments/results
until grep -q "^\[final\] done" $R/final_queue.log 2>/dev/null; do sleep 20; done
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done
for U in 3 6; do
  echo "[alpha] $(date +%H:%M:%S) E4 utterance $U with signal gain"
  python experiments/exp_nullspace_survival.py --n-files 20 --spk-offset 20 --utt-index $U \
    --t 50,100,200,300 --null-t 100,200 --seeds 2026,7 --sampler ddim \
    --purifier-ckpt checkpoints/purifier_v3.pt \
    --out $R/nullspace_survival_20spk_u${U}_alpha.csv > $R/surv20_u${U}_alpha.log 2>&1
  grep -v -i "warn\|tensorflow\|oneDNN" $R/surv20_u${U}_alpha.log | sed -n '/MEAN over/,$p'
done
echo "[alpha] done"
