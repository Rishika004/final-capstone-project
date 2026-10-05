#!/usr/bin/env bash
# ============================================================================
# THE MISSING CELL: mismatch trade at the LOUD budget WITH segment crafting.
#
# run_rebase_mismatch_m30.sh (RESULTS 26) replicated the claim on the corrected
# base with POOLED crafting: 6 of 12 cells help after Holm. But every cloning
# number in the paper uses --seg-partials (fix P0), and the one run that used it
# (m20) showed nothing -- while also changing the budget. So budget and crafting
# are confounded and the paper cannot yet say the mismatch claim holds under the
# crafting it actually ships.
#
# This run is the m30 script with exactly ONE change: --seg-partials on. Same
# base, same calibrations (already on disk from the m30 run; step 0 skips), same
# arms, same seeds. Outputs are suffixed _seg so nothing is overwritten.
#
#   holds  -> one recipe for the whole paper. Done.
#   fails  -> seg-partials trades the mismatch benefit for the d-vector fix, and
#             the paper reports that trade explicitly; it does not pick the
#             recipe that flatters each claim.
#
# Run:  bash experiments/run_rebase_mismatch_m30_seg.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_m30_seg.csv
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[m30seg] $(date +%H:%M:%S) $*"; }

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
      > $R/m30seg_calib_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------------ 1. craft
# --seg-partials ON. The only difference from run_rebase_mismatch_m30.sh.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --seg-partials --out $OUT"

run() {
  tag=$1; shift
  if grep -q "^$tag," $OUT 2>/dev/null; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/m30seg_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run plain_tc_m30_seg    --variant plain --margin-db 30 --snr-floor 15
run nsT_tc_m30_seg      --variant plain --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run full_tc_m30_seg     --variant full --vocoder phase --margin-db 30 --snr-floor 15
run fullnsT_tc_m30_seg  --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200

# BOTH budgets in one pass, so the 2x2 is scored against identical thresholds.
# eval_saved reads saved wavs by tag -- no re-crafting, so the m20 half is nearly free.
TAGS=plain_tc_m30_seg,nsT_tc_m30_seg,full_tc_m30_seg,fullnsT_tc_m30_seg

# ------------------------------------------------------------- 2. block (c)
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    O=$R/protect_foreign_seg_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing -- refusing the legacy file"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "MISMATCHED: $TAG t=$T, corrected thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      > $R/m30seg_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------- 3. block (a)
# NO purifier. Without this the whole reading collapses: a perturbation that is
# merely sturdier would win here too, and the claim is specifically that it does not.
if [ ! -f $R/control_nopurifier_seg.csv ]; then
  CTHR=$(thr_of "$R/thresholds_clean_xsess_tc.csv")
  say "CONTROL: no purifier, thresholds $CTHR"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --thresholds "$CTHR" --out $R/control_nopurifier_seg.csv \
    > $R/m30seg_control.log 2>&1
  say "done (exit $?)"
fi

say "DONE. Read the 2x2 with: python experiments/analyse_mismatch_final.py"
say "This is the recipe the paper ships. Read it before typesetting rows 4-5."
