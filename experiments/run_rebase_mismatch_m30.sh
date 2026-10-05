#!/usr/bin/env bash
# ============================================================================
# REBASE THE MISMATCH TRADE **AT THE BUDGET IT WAS PUBLISHED AT**.
#
# WHY THIS RUN EXISTS
# -------------------
# run_rebase_mismatch.sh (2026-09-20 overnight) was supposed to move the claim
#   (a) no purifier  -> constraint LOSES
#   (b) matched      -> breaks even
#   (c) mismatched   -> constraint WINS (7 of 12 after Holm)
# onto the corrected base. It changed FOUR things it was meant to change
# (dev-clean->test-clean, v3->v7, same-session->cross-session, 20->29 speakers)
# and TWO it was not:
#
#   1. THE BUDGET.  Every old foreign/control CSV holds only *_m30_f15* arms.
#      That run used m20_f20. The script justified it as "the loud budget
#      saturates and cannot distinguish arms" -- true of the BINARY rate, but
#      this claim has always been measured on the CONTINUOUS margin, where
#      saturation does not apply. So the claim was rebased onto a budget it was
#      never measured at.
#   2. --seg-partials.  Only DvecEnc has embed_partials, so the flag changes the
#      d-vector term only -- but that term feeds the shared LSE loss, so it
#      reallocates perturbation budget away from WavLM/ECAPA, which are the two
#      encoders that carried the claim.
#
# Result of that run: 4 of 12 significant instead of 7, and all four on the
# d-vector with POSITIVE deltas (constraint protects WORSE). WavLM effects
# shrank 4-5x and stopped surviving Holm. That is a sign flip, and the pre-
# registered instruction was to investigate it rather than absorb it.
#
# THIS RUN CHANGES ONLY THE FOUR INTENDED THINGS.
#   budget    m30_f15, as published
#   crafting  POOLED -- no --seg-partials, as published
#   base      test-clean / v7 / cross-session / 29 speakers / shared selector
#
# READ IT AS A 2x2 AGAINST THE m20 RUN, not as a replacement:
#   holds here, fails at m20  -> the claim is real and BUDGET-CONDITIONAL. Say so
#                                in the paper; do not quote it unconditionally.
#   fails here too            -> the claim was a property of the old base. Drop it.
#
# STEP 0 ALSO FIXES A DEFECT IN THE PREVIOUS RUN. Its `[ ! -f "$CAL" ]` guard
# found the LEGACY thresholds_diffwave_t{1,3}[_sc09].csv -- files with no
# cross_session/subset column, d-vector EER 0.0009, i.e. exactly the same-session
# optimism fix S1 exists to remove -- and reused them. Paired deltas are immune
# (the threshold cancels in a difference) but every binary rate in block (c) is
# wrong. We recalibrate on the corrected base and rescore BOTH budgets against
# the same calibration, so the 2x2 is scored identically throughout.
#
# Run:  bash experiments/run_rebase_mismatch_m30.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_m30.csv
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[m30] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }

# Threshold triples are pulled by a helper script rather than an inline heredoc:
# nesting a heredoc inside a function inside this file is how the previous
# version of this script got mangled.
thr_of() { python experiments/_thr_triple.py "$1"; }

THR=$(thr_of "$THRFILE")
say "matched thresholds (v7/ddim t=50, cross-session, $SUBSET): $THR"

# ------------------------------------------------- 0. honest foreign calibration
# Named _xsess_tc so it can never be confused with the legacy file again. The
# guard below tests for THIS name, not the legacy one.
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip calib $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] && { say "skip calib $TAG t=$T"; continue; }
    say "CALIBRATE $TAG t=$T on $SUBSET, cross-session, $NFILES speakers"
    python experiments/calibrate_thresholds.py --channel diffwave \
      --foreign-ckpt "$CKPT" --t-diffuse $T --subset $SUBSET --cross-session \
      --n-spk $NFILES --n-utt 4 --tag ${TAG}_xsess_tc \
      > $R/m30_calib_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------------ 1. craft
# NO --seg-partials. This is the published crafting path, deliberately.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --out $OUT"

run() {
  tag=$1; shift
  if grep -q "^$tag," $OUT 2>/dev/null; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/m30_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run plain_tc_m30    --variant plain --margin-db 30 --snr-floor 15
run nsT_tc_m30      --variant plain --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run full_tc_m30     --variant full --vocoder phase --margin-db 30 --snr-floor 15
run fullnsT_tc_m30  --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200

# BOTH budgets in one pass, so the 2x2 is scored against identical thresholds.
# eval_saved reads saved wavs by tag -- no re-crafting, so the m20 half is nearly free.
TAGS=plain_tc_m30,nsT_tc_m30,full_tc_m30,fullnsT_tc_m30,plain_tc_m20,nsT_tc_m20,full_tc_m20,fullnsT_tc_m20

# ------------------------------------------------------------- 2. block (c)
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    O=$R/protect_foreign_both_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing -- refusing the legacy file"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "MISMATCHED: $TAG t=$T, corrected thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      > $R/m30_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------- 3. block (a)
# NO purifier. Without this the whole reading collapses: a perturbation that is
# merely sturdier would win here too, and the claim is specifically that it does not.
if [ ! -f $R/control_nopurifier_both.csv ]; then
  CTHR=$(thr_of "$R/thresholds_clean_xsess_tc.csv")
  say "CONTROL: no purifier, thresholds $CTHR"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --thresholds "$CTHR" --out $R/control_nopurifier_both.csv \
    > $R/m30_control.log 2>&1
  say "done (exit $?)"
fi

say "DONE. Read the 2x2 with: python experiments/analyse_budget_2x2.py"
say "Do NOT write the claim into the paper until that 2x2 is read."
