#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"
# ============================================================================
# THE ARM RESULTS 47e MADE MANDATORY.
#
# Control A (RESULTS 47) compared matched against held-out at the POOLED
# budget, because `full_m30_seg_nodvec` did not exist.  It returned OUTCOME
# A-2: the d-vector reversal disappears and flips sign when the d-vector is
# held out of the crafting ensemble.
#
# PREREG_council5_controls.md A.2 declared, before the control was run:
#     "If the control contradicts the matched result, the seg version becomes
#      mandatory."
# It contradicts.  This script crafts that arm.
#
# WHAT IS CRAFTED, AND WHAT IS NOT
# --------------------------------
# `fullnsT_m30_seg_nodvec` (the CONSTRAINT arm, held out, segment partials) has
# been on disk since 2026-09-20, 29 speakers, run_p0_segments.sh craft B.
# Only its unconstrained counterpart is missing.  So ONE arm is crafted here
# and paired with the stored one.
#
# That pairing was checked rather than assumed --
# experiments/check_seg_nodvec_repro.py, RESULTS 50:
#   HEAD vs HEAD      worst 0.0328 on one speaker
#   stored v1 vs HEAD worst 0.0323 on the same speaker
# The crafting loop is NOT run-to-run deterministic at a fixed seed, and the
# stored-vs-HEAD difference is the SAME SIZE as HEAD against itself.  So
# re-crafting the stored arm would buy nothing: two HEAD runs differ as much as
# HEAD differs from v1.  One arm, as 47e budgeted, for a reason 47e did not
# know.
#
# FLAGS: identical to run_p0_segments.sh craft B except that --lam-null and
# --null-t are absent, which is the one variable.  Compare:
#   craft B:  --variant full --vocoder phase --margin-db 30 --snr-floor 15 \
#             --lam-null 10 --null-t 200   --craft-encoders no-dvec
#   here:     --variant full --vocoder phase --margin-db 30 --snr-floor 15 \
#                                            --craft-encoders no-dvec
#
# PREDICTION, recorded before the arm is crafted (RESULTS 49e):
#   crafting mode moved the matched cells by 0.0072 on average, max 0.0192,
#   with zero sign flips.  So the seg held-out grid should reproduce 47's
#   pooled held-out grid to within about 0.01 and MUST NOT flip a sign.
#   A sign flip means something other than budget or crafting mode is
#   operating, and 47's reading would need re-examination.
#
# Run:  bash experiments/run_heldout_seg.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_seg_heldout.csv
PREREG=paper/PREREG_council5_controls.md
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[heldout-seg] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
thr_of() { python experiments/_thr_triple.py "$1"; }
THR=$(thr_of "$THRFILE")
say "matched thresholds: $THR"

COMMON="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,200 --thr-aware --weights 1,1,1 --seg-partials \
 --craft-encoders no-dvec --prereg $PREREG"

TAG=full_m30_seg_nodvec
if skip_guard "$TAG" "$OUT"; then
  say "skip $TAG"
else
  say "CRAFT $TAG -- the unconstrained held-out arm at the shipping budget"
  python experiments/exp_purifier_aware.py $COMMON \
    --variant full --vocoder phase --margin-db 30 --snr-floor 15 \
    --out "$OUT" --tag "$TAG" > $R/heldout_seg_craft.log 2>&1
  say "done (exit $?)"
fi

# ---------------------------------------------------------------- evaluate
# Both arms through the four foreign channels, against the SAME calibrations
# Control A used, so the seg grid is directly comparable to the pooled one.
TAGS=full_m30_seg_nodvec,fullnsT_m30_seg_nodvec

for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; NAME="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip $NAME (checkpoint absent)"; continue; }
  for T in 1 3; do
    O=$R/heldout_seg_c2_${NAME}_t${T}.csv
    [ -f "$O" ] && { say "skip eval $NAME t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${NAME}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing -- refusing the legacy file"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "EVAL $NAME t=$T, thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      --prereg $PREREG > $R/heldout_seg_${NAME}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

say "DONE. Read with: python experiments/analyse_heldout_seg.py"
