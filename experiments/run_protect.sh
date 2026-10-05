#!/usr/bin/env bash
# Replicate the NEGATIVE result on a second utterance per speaker.
# Survival now has a two-utterance replication (RESULTS 13c); protection does not,
# and protection is the paper's core claim. Four variants carry the argument:
#   plain     baseline additive crafting
#   nsT       pass-through constraint ALONE      -> reported to HURT
#   full      purifier-aware EOT crafting        -> reported to WORK
#   fullnsT   EOT + pass-through constraint      -> reported to ADD NOTHING
# Settings copied exactly from main_table_v3.csv so the two utterances are
# comparable: margin 30 dB, SNR floor 15, threshold-aware, DDIM, phase vocoder
# for crafting / Griffin-Lim for the attacker, 200 steps, weights 1,1,1.
cd "$(dirname "$0")/.."
R=experiments/results
until grep -q "^\[alpha\] done" $R/alpha_queue.log 2>/dev/null; do sleep 20; done
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done

COMMON="--dataset librispeech --n-files 20 --spk-offset 20 --utt-index 6 \
 --margin-db 30 --snr-floor 15 --thr-aware --sampler ddim --vocoder phase \
 --attacker-vocoder gl --steps 200 --weights 1,1,1 --t-diffuse 50 \
 --eval-t 50,100,200 --purifier-ckpt checkpoints/purifier_v3.pt \
 --out $R/main_table_v3_u6.csv"

run () {  # name, extra args
  echo "[protect] $(date +%H:%M:%S) $1"
  python experiments/exp_purifier_aware.py $COMMON --tag "$1" $2 \
    >> $R/protect_u6.log 2>&1
}
run plain_m30_f15        "--variant plain"
run nsT_m30_f15_l10      "--variant plain --lam-null 10 --null-t 200"
run full_m30_f15         "--variant full"
run fullnsT_m30_f15_l10  "--variant full --lam-null 10 --null-t 200"
echo "[protect] done"
