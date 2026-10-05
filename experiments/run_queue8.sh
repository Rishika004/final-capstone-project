#!/usr/bin/env bash
# Null-space, noise-matched: (1) survival test with noise-matched directions,
# (2) 3-speaker sweep of NS-PGD (plain crafting + noise-matched regulariser),
# (3) 20-speaker runs of the best setting. Chained behind queue 7.
cd "$(dirname "$0")/.."
while ! grep -q "all done" experiments/results/queue7.log 2>/dev/null; do sleep 30; done
CK=checkpoints/purifier_v3.pt
say() { echo "[queue8] $(date +%H:%M:%S) $*"; }

say "E1b noise-matched survival (v3/ddim)"
python experiments/exp_nullspace_survival.py --n-files 5 --t 50,100,200,300 --purifier-ckpt $CK --sampler ddim \
   --null-t 100,200 --out experiments/results/nullspace_survival_v3_ddim_matched.csv \
   > experiments/results/nullspace_survival_v3_ddim_matched.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" experiments/results/nullspace_survival_v3_ddim_matched.log | sed -n '/MEAN/,$p'

THR=$(python - <<'EOF'
import pandas as pd
d = pd.read_csv("experiments/results/thresholds_diffpure_gl_t50_ddim_v3.csv").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.3f},{d['ECAPA-TDNN']:.3f},{d['Resemblyzer d-vector']:.3f}")
EOF
)
BASE="--dataset librispeech --spk-offset 20 --utt-index 3 --steps 200 \
 --purifier-ckpt $CK --sampler ddim --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1"
run() { # out n tag args
  out=$1; n=$2; tag=$3; shift 3
  say "start $tag (n=$n)"
  python experiments/exp_purifier_aware.py $BASE --out $out --n-files $n --tag $tag "$@" \
      > experiments/results/q8_${tag}.log 2>&1
  say "done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/q8_${tag}.log | grep "t= 50\|t=100\|t=200\|quality"
}
S=experiments/results/purifier_aware.csv
M=experiments/results/main_table_v3.csv
# 3-speaker sweep: lambda x noise level, at the decisive budget
run $S 3 ns_m30_f15_l3_t200  --variant plain --margin-db 30 --snr-floor 15 --lam-null 3  --null-t 200
run $S 3 ns_m30_f15_l10_t200 --variant plain --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run $S 3 ns_m30_f15_l10_t100 --variant plain --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 100
run $S 3 ns_m30_f15_l30_t200 --variant plain --margin-db 30 --snr-floor 15 --lam-null 30 --null-t 200
# 20-speaker: two settings (pick-best happens offline; both are cheap, ~25 min each)
run $M 20 nsT_m30_f15_l10  --variant plain --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run $M 20 nsT_m20_f20_l10  --variant plain --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200
say "all done"
