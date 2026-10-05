#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "[mc] $(date +%H:%M:%S) quality"
python experiments/eval_quality_nr.py --n-files 20 --transforms mcadams \
   --additive-tags full_m30_f15,fullnsT_m30_f15_l10 \
   --out experiments/results/quality_mcadams.csv > experiments/results/quality_mcadams.log 2>&1
grep -v -i "tensorflow\|oneDNN\|warn" experiments/results/quality_mcadams.log | sed -n '/NO-REFERENCE/,$p'
echo "[mc] $(date +%H:%M:%S) protection"
python experiments/exp_parametric_probe.py --n-files 20 --transforms mcadams \
   --t 50,200 --seeds 2026,7 --save-wavs 2 \
   --out experiments/results/parametric_mcadams.csv > experiments/results/parametric_mcadams.log 2>&1
grep -v -i "tensorflow\|oneDNN\|warn" experiments/results/parametric_mcadams.log | sed -n '/SURVIVAL/,$p'
echo "[mc] $(date +%H:%M:%S) all done"
