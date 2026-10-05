#!/usr/bin/env bash
# Final attacker = purifier v3 (2-D residual UNet) + DDIM reverse process.
# 1. channel calibration (t=50/100/200)   2. E1 null-space survival
# 3. transfer: files crafted against v2/DDPM evaluated against v3/DDIM
# 4. main table on 20 held-out speakers, four crafting methods x two budgets
cd "$(dirname "$0")/.."
while ! grep -q "all done" experiments/results/queue6.log 2>/dev/null; do sleep 20; done
CK=checkpoints/purifier_v3.pt
say() { echo "[queue7] $(date +%H:%M:%S) $*"; }

for T in 50 100 200; do
  say "calibration v3/ddim t=$T"
  python experiments/calibrate_thresholds.py --channel diffpure_gl --t-diffuse $T --purifier-ckpt $CK --sampler ddim \
     --spk-offset 20 --n-spk 20 --n-utt 3 --tag v3 > experiments/results/calib_diffpure_gl_t${T}_ddim_v3.log 2>&1
  grep "EER" experiments/results/calib_diffpure_gl_t${T}_ddim_v3.log
done

say "E1 null-space survival v3/ddim"
python experiments/exp_nullspace_survival.py --n-files 5 --t 50,100,200,300 --purifier-ckpt $CK --sampler ddim \
   --out experiments/results/nullspace_survival_v3_ddim.csv > experiments/results/nullspace_survival_v3_ddim.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/nullspace_survival_v3_ddim.log | sed -n '/MEAN/,$p'

say "transfer: v2-crafted files vs v3/ddim"
python experiments/eval_saved.py --purifier-ckpt $CK --sampler ddim --t 0,50,100,200 \
   --out experiments/results/eval_saved_v3_ddim.csv > experiments/results/eval_saved_v3_ddim.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/eval_saved_v3_ddim.log | sed -n '/PROTECTION/,$p'

# ---- main table (thresholds: v3/ddim t=50 calibration is read from its csv)
THR=$(python - <<'EOF'
import pandas as pd
d = pd.read_csv("experiments/results/thresholds_diffpure_gl_t50_ddim_v3.csv").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.3f},{d['ECAPA-TDNN']:.3f},{d['Resemblyzer d-vector']:.3f}")
EOF
)
say "main table thresholds (v3/ddim t=50): $THR"
BASE="--dataset librispeech --spk-offset 20 --utt-index 3 --n-files 20 --steps 200 \
 --purifier-ckpt $CK --sampler ddim --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --out experiments/results/main_table_v3.csv"
run() {
  tag=$1; shift
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" > experiments/results/main3_${tag}.log 2>&1
  say "done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/main3_${tag}.log | grep "t= 50\|t=100\|t=200\|quality"
}
run plain_m30_f15     --variant plain --margin-db 30 --snr-floor 15
run null_m30_f15      --variant plain --margin-db 30 --snr-floor 15 --lam-null 3.0
run full_m30_f15      --variant full --vocoder phase --margin-db 30 --snr-floor 15
run fullnull_m30_f15  --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 3.0
run plain_m20_f20     --variant plain --margin-db 20 --snr-floor 20
run null_m20_f20      --variant plain --margin-db 20 --snr-floor 20 --lam-null 3.0
run full_m20_f20      --variant full --vocoder phase --margin-db 20 --snr-floor 20
say "all done"
