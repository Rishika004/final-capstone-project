#!/usr/bin/env bash
# ============================================================================
# #5 SENSITIVITY CHECK -- prereg section 2.2. NOT the result.
#
# The primary scoring of the 12 cells uses thresholds calibrated on TEST-CLEAN,
# which is the whole point: the operating point must not be set on the speakers
# it scores. But that choice could itself carry the verdict, so the
# pre-registration fixes a check: re-score the SAME crafted files against
# thresholds calibrated on dev-clean itself.
#
#   same verdict      -> the primary scoring stands
#   different verdict -> the verdict is "THRESHOLD-DEPENDENT" and NEITHER
#                        scoring is reported as the result
#
# Nothing is re-crafted here. Only the operating point changes.
#
# RUN THIS ONLY AFTER run_devclean_replication.sh has finished -- it shares the
# GPU, and prereg section 7.6 forbids a second GPU job beside the crafting.
#
# Run:  bash experiments/run_devclean_sensitivity.sh
# Read: python experiments/analyse_devclean.py --suffix _dcthr
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

SUBSET=dev-clean
NFILES=31
R=experiments/results
PREREG=paper/PREREG_05_devclean_replication.md
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
TAGS=full_dc_m30_seg_bpda,fullnsT_dc_m30_seg_bpda
say() { echo "[dc5-sens] $(date +%H:%M:%S) $*"; }
thr_of() { python experiments/_thr_triple.py "$1"; }

[ -f $R/main_table_dc_m30_seg_bpda.csv ] || {
  say "FATAL: the primary run has not been assembled. Run"
  say "       run_devclean_replication.sh first."; exit 1; }

# -------------------------------------------- 1. calibrate ON DEV-CLEAN
# The thing the primary scoring deliberately avoids. --dump-scores so the same
# minDCF / DET / speaker-bootstrap treatment as council #19 is available for
# these channels too, at no extra GPU cost once the embeddings are computed.
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "FATAL: $CKPT absent"; exit 1; }
  for T in 1 3; do
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_dc.csv
    [ -f "$CAL" ] && { say "skip calib $TAG t=$T (exists)"; continue; }
    say "CALIBRATE ON DEV-CLEAN: $TAG t=$T, $NFILES speakers, cross-session"
    python experiments/calibrate_thresholds.py --channel diffwave \
      --foreign-ckpt "$CKPT" --t-diffuse $T --subset $SUBSET --cross-session \
      --n-spk $NFILES --n-utt 4 --dump-scores --tag ${TAG}_xsess_dc \
      > $R/dc5_sens_calib_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------- 2. re-score the same 12 cells
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  for T in 1 3; do
    O=$R/dc5_foreign_${TAG}_t${T}_dcthr.csv
    [ -f "$O" ] && { say "skip $TAG t=$T (exists)"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_dc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "RE-SCORE $TAG t=$T against DEV-CLEAN thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      --prereg $PREREG > $R/dc5_sens_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

say "DONE. Compare the two verdicts:"
say "  python experiments/analyse_devclean.py               # primary"
say "  python experiments/analyse_devclean.py --suffix _dcthr   # sensitivity"
say "If they differ, the verdict is THRESHOLD-DEPENDENT and neither is the result."
