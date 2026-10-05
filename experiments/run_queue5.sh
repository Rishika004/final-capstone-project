#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while tasklist | grep -qi "python.exe"; do sleep 15; done
echo "[queue5] $(date +%H:%M:%S) start E1 strong purification"
python experiments/exp_nullspace_survival.py --n-files 5 --t 200,300,500 > experiments/results/nullspace_survival_strong.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/nullspace_survival_strong.log | tail -8
echo "[queue5] $(date +%H:%M:%S) start eval_saved"
python experiments/eval_saved.py --t 0,50,200 > experiments/results/eval_saved.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/eval_saved.log | tail -30
echo "[queue5] all done"
