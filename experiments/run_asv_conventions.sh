#!/usr/bin/env bash
# ============================================================================
# Close council #19 on the channels THE PAPER ACTUALLY REPORTS.
#
# 42 gave EER + minDCF + speaker-bootstrap CIs for the clean channel, and 42g
# added the four foreign channels -- but those were calibrated on DEV-CLEAN,
# because that is the cohort the #5 sensitivity run needed. 45g then showed
# dev-clean is the harder cohort (ECAPA EER 4.84 % against 0.86 %), so those
# numbers are NOT a substitute for the operating points the main table is
# scored against.
#
# This run calibrates the SAME channels on TEST-CLEAN with --dump-scores, which
# is the only missing ingredient: minDCF, DET and a speaker-cluster bootstrap
# all need raw trial scores, and none of the frozen files has them.
#
# NOTHING FROZEN IS OVERWRITTEN. Every output carries a `_v2` tag, exactly as
# the clean-channel re-run did, so `thresholds_*_xsess_tc.csv` -- the files the
# paper's numbers were scored against -- are untouched. The recomputed
# thresholds are then COMPARED against the frozen ones, which doubles as a
# reproducibility check: the clean channel reproduced to four decimals (42a).
#
# Channels, matching run_testclean_pipeline.sh and run_rebase_mismatch_m30_seg.sh
# flag for flag:
#   diffpure_gl (ours, v7, DDIM) at t = 50, 100, 200   -- the matched upper bound
#   DiffWave LibriSpeech and SC09 at t = 1, 3          -- the mismatch cells
#
# Run:  bash experiments/run_asv_conventions.sh
# Read: python experiments/analyse_asv.py
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NSPK=29
R=experiments/results
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[asv] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }

# ------------------------------------------- 1. ours / DDIM, the matched channel
for T in 50 100 200; do
  OUT=$R/thresholds_diffpure_gl_t${T}_ddim_v7_xsess_tc_v2.csv
  [ -f "$OUT" ] && { say "skip diffpure t=$T (exists)"; continue; }
  say "calibrate diffpure_gl/DDIM t=$T on $SUBSET, cross-session, $NSPK spk"
  python experiments/calibrate_thresholds.py --channel diffpure_gl --t-diffuse $T \
    --purifier-ckpt $CK --sampler ddim --subset $SUBSET --cross-session \
    --n-spk $NSPK --n-utt 4 --dump-scores --tag v7_xsess_tc_v2 \
    > $R/asv_calib_diffpure_t${T}.log 2>&1
  say "done (exit $?)"
done

# ------------------------------------------------ 2. the foreign mismatch cells
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "FATAL: $CKPT absent"; exit 1; }
  for T in 1 3; do
    OUT=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc_v2.csv
    [ -f "$OUT" ] && { say "skip diffwave $TAG t=$T (exists)"; continue; }
    say "calibrate DiffWave-$TAG t=$T on $SUBSET, cross-session, $NSPK spk"
    python experiments/calibrate_thresholds.py --channel diffwave \
      --foreign-ckpt "$CKPT" --t-diffuse $T --subset $SUBSET --cross-session \
      --n-spk $NSPK --n-utt 4 --dump-scores --tag ${TAG}_xsess_tc_v2 \
      > $R/asv_calib_diffwave_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------- 3. reproducibility check
say "comparing the recomputed thresholds against the frozen ones"
python experiments/check_calibration_drift.py || {
  say "FATAL: a recomputed threshold does not match the frozen file."
  say "       That is a reproducibility failure, not a rounding difference."
  exit 1; }

say "DONE. Read with: python experiments/analyse_asv.py"
