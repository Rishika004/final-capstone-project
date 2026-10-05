#!/usr/bin/env bash
# Characterise purifier v3 (2-D residual UNet) and re-run the null-space test.
cd "$(dirname "$0")/.."
CK=checkpoints/purifier_v3.pt
say() { echo "[queue6] $(date +%H:%M:%S) $*"; }

say "start eval_saved v3 (perturbation removal + protection on saved 20-spk files)"
python experiments/eval_saved.py --purifier-ckpt $CK --t 0,50,100,200 --out experiments/results/eval_saved_v3.csv \
   > experiments/results/eval_saved_v3.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/eval_saved_v3.log | sed -n '/PROTECTION/,$p'

for T in 50 100 200; do
  say "start calibration v3 t=$T"
  python experiments/calibrate_thresholds.py --channel diffpure_gl --t-diffuse $T --purifier-ckpt $CK \
     --spk-offset 20 --n-spk 20 --n-utt 3 --tag v3 > experiments/results/calib_diffpure_gl_t${T}_v3.log 2>&1
  grep "EER" experiments/results/calib_diffpure_gl_t${T}_v3.log
done

say "start E1 null-space survival v3"
python experiments/exp_nullspace_survival.py --n-files 5 --t 50,100,200,300 --purifier-ckpt $CK \
   > experiments/results/nullspace_survival_v3.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/nullspace_survival_v3.log | sed -n '/MEAN/,$p'
say "all done"
