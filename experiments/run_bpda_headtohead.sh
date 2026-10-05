#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"   # W0: refuse to skip rows written by other code
# ============================================================================
# HEAD-TO-HEAD: BPDA+EOT vs exact-gradient EOT, everything else identical.
#
# The council's ML and security reviewers made the same point: the paper's
# framing -- "De-AntiFake 5.4 and VocalBridge V-E concluded adaptive protection
# fails; with exact gradients it works" -- is not earned until BPDA is run on
# OUR purifier under OUR conditions. This script is run_rebase_mismatch_m30_seg.sh
# with ONE change: --grad-estimator bpda (real purifier forward, identity
# backward). Same base, budget, crafting, seeds, thresholds, EOT samples, steps.
# Only the two EOT arms are crafted (plain/nsT do not touch the purifier).
# Outputs carry a _bpda suffix. Compare against the _seg files:
#   protection under matched / foreign purifiers   (eval_saved, this script)
#   then DiffVC cloning                             (queued separately)
# Smoke-tested on one speaker, six steps, 2026-09-22 15:2x.
#
# Run:  bash experiments/run_bpda_headtohead.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_m30_seg_bpda.csv
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[bpda] $(date +%H:%M:%S) $*"; }

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
      > $R/bpda_calib_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------------ 1. craft
# --grad-estimator bpda is the ONLY difference from run_rebase_mismatch_m30_seg.sh.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --seg-partials --grad-estimator bpda --out $OUT"

run() {
  tag=$1; shift
  if skip_guard "$tag" "$OUT"; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/bpda_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run full_tc_m30_seg_bpda     --variant full --vocoder phase --margin-db 30 --snr-floor 15
run fullnsT_tc_m30_seg_bpda  --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200

# BOTH budgets in one pass, so the 2x2 is scored against identical thresholds.
# eval_saved reads saved wavs by tag -- no re-crafting, so the m20 half is nearly free.
TAGS=full_tc_m30_seg_bpda,fullnsT_tc_m30_seg_bpda

# ------------------------------------------------------------- 2. block (c)
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "skip $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    O=$R/protect_foreign_bpda_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing -- refusing the legacy file"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "MISMATCHED: $TAG t=$T, corrected thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      > $R/bpda_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------------------- 3. block (a)
# NO purifier. Without this the whole reading collapses: a perturbation that is
# merely sturdier would win here too, and the claim is specifically that it does not.
if [ ! -f $R/control_nopurifier_bpda.csv ]; then
  CTHR=$(thr_of "$R/thresholds_clean_xsess_tc.csv")
  say "CONTROL: no purifier, thresholds $CTHR"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --thresholds "$CTHR" --out $R/control_nopurifier_bpda.csv \
    > $R/bpda_control.log 2>&1
  say "done (exit $?)"
fi

say "DONE. Read the 2x2 with: python experiments/analyse_bpda_headtohead (to write)"
say "Compare _bpda against _seg before the 'reverses De-AntiFake' sentence is written."
