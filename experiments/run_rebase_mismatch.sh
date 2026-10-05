#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"   # W0: refuse to skip rows written by other code
# ============================================================================
# REBASE THE MISMATCH TRADE -- the paper's actual contribution -- onto the
# corrected evaluation base. This is the last claim still on the old protocol.
#
# THE CLAIM BEING REBASED
# -----------------------
# "Purification-invariance is a trade." Read as three blocks that only mean
# something together:
#   (a) NO purifier      -> the constraint LOSES (0/20 on both strong encoders)
#   (b) MATCHED purifier -> it breaks even (the EOT gradients already cover it)
#   (c) MISMATCHED       -> it WINS (7 of 12 cells after Holm)
# Block (a) is what rules out "they just built a sturdier perturbation", so it is
# not optional -- a sturdier perturbation would win in all three.
#
# WHAT IT NEEDS, AND WHY IT IS THE EXPENSIVE ONE
#   4 crafting arms x 29 speakers      (plain / nsT / full / fullnsT, m20 budget)
#   + evaluation through 2 foreign purifiers at 2 strengths
#   + the no-purifier control
# The m20 (tight) budget is deliberate: the loud budget SATURATES and cannot
# distinguish arms, which is right for a headline and useless for an ablation.
#
# EXPECTATION, WRITTEN BEFORE THE RUN
# Absolute protection rates will FALL -- cross-session thresholds are lower, so
# every encoder accepts more readily. The PAIRED results should not move: the
# threshold cancels in a difference, which is exactly what fix S2 demonstrated
# (rates changed, p-values identical to 4 decimals). If a paired result moves
# materially here, investigate it; do not absorb it.
#
# Run:  bash experiments/run_rebase_mismatch.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_m20.csv
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[mm] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
THR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$THRFILE").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
say "thresholds (v7/ddim t=50, cross-session, $SUBSET): $THR"

# ---------------------------------------------------------------- 1. craft
# --seg-partials is ON: fix P0 showed the pooled d-vector objective was ~0.13
# cosine away from the scoring function. Every arm uses it, so the ablation
# compares arms rather than comparing crafting code.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --seg-partials --out $OUT"

run() {
  tag=$1; shift
  if skip_guard "$tag" "$OUT"; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/mm_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run plain_tc_m20    --variant plain --margin-db 20 --snr-floor 20
run nsT_tc_m20      --variant plain --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200
run full_tc_m20     --variant full --vocoder phase --margin-db 20 --snr-floor 20
run fullnsT_tc_m20  --variant full --vocoder phase --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200

TAGS=plain_tc_m20,nsT_tc_m20,full_tc_m20,fullnsT_tc_m20

# ---------------------------------------------------------------- 2. block (c)
# MISMATCHED: the same saved perturbations pushed through purifiers we never
# trained. Each foreign channel is scored at ITS OWN calibration (fix S2) --
# reusing one channel's threshold for another is not allowed.
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    O=$R/protect_foreign_tc_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}$([ "$TAG" = sc09 ] && echo _sc09).csv
    if [ ! -f "$CAL" ]; then
      say "calibrating $TAG t=$T on $SUBSET (cross-session)"
      python experiments/calibrate_thresholds.py --channel diffwave \
        --foreign-ckpt "$CKPT" --t-diffuse $T --subset $SUBSET --cross-session \
        --n-spk $NFILES --n-utt 4 --tag tc_${TAG} \
        > $R/mm_calib_${TAG}_t${T}.log 2>&1
      CAL=$R/thresholds_diffwave_t${T}_tc_${TAG}.csv
    fi
    FTHR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$CAL").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
    say "MISMATCHED: $TAG t=$T, thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      > $R/mm_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ---------------------------------------------------------------- 3. block (a)
# NO purifier. The control that makes the whole reading work: if the constraint
# also won here, the result would be generic robustness, not mismatch insurance.
if [ ! -f $R/control_nopurifier_tc.csv ]; then
  CTHR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$R/thresholds_clean_xsess_tc.csv").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
  say "CONTROL: no purifier, thresholds $CTHR"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --thresholds "$CTHR" --out $R/control_nopurifier_tc.csv \
    > $R/mm_control.log 2>&1
  say "done (exit $?)"
fi

say "MISMATCH REBASED. The paper now sits on ONE evaluation base."
say "Read with: python experiments/analyse_mismatch.py (repoint it at the _tc files)"
