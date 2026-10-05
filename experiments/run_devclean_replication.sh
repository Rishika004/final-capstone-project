#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"   # W0: refuse to skip rows written by other code
# ============================================================================
# #5 -- OUT-OF-SAMPLE REPLICATION OF C2 ON DEV-CLEAN.  PLAN_V3 W3-W5.
#
# This script executes `paper/PREREG_05_devclean_replication.md` and nothing
# else. Read that file first: it fixes the 12 cells, the primary endpoint, the
# estimator, and the outcome->action table, and it was committed before any
# dev-clean file was crafted. Its content hash lands in every output row as
# `prereg_sha`.
#
# WHAT IS BEING ASKED
#   C2 says the purification-invariance constraint helps under purifier
#   MISMATCH, with a per-encoder pattern that reverses on the d-vector. That is
#   currently one sample of 29 test-clean speakers scored against thresholds
#   calibrated on those same speakers. This run asks whether the pattern shows
#   up in 31 speakers we have never crafted against, scored against thresholds
#   calibrated on a disjoint cohort.
#
# WHAT MUST NOT HAPPEN (prereg section 7)
#   * no flag in prereg section 3 may change
#   * SKIPGUARD_FORCE_SKIP must NOT be used for the crafting stages
#   * NO OTHER GPU JOB may run beside this one
#   * the thresholds are the TEST-CLEAN cross-session calibrations; a
#     dev-clean-calibrated scoring is a pre-specified sensitivity check,
#     reported separately, never as the result
#
# Run:  bash experiments/run_devclean_replication.sh
# Read: python experiments/analyse_devclean.py
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=dev-clean
NFILES=31
R=experiments/results
PREREG=paper/PREREG_05_devclean_replication.md
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[dc5] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ]     || { say "FATAL: $CK missing"; exit 1; }
[ -f "$PREREG" ] || { say "FATAL: $PREREG missing -- this run has no pre-registration"; exit 1; }

# Thresholds: calibrated on TEST-CLEAN, per channel (prereg section 2.2). The
# crafting loss uses the matched v7 t=50 triple, exactly as the test-clean run
# did; the mismatch cells are scored against their own foreign calibrations
# further down.
THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
thr_of() { python experiments/_thr_triple.py "$1"; }
THR=$(thr_of "$THRFILE")
say "crafting thresholds (test-clean v7/ddim t=50): $THR"
say "prereg sha: $(python utils/provenance.py --stored /dev/null x >/dev/null 2>&1; git hash-object "$PREREG" | cut -c1-12)"

# ------------------------------------------------------------------ 1. craft
# prereg section 3: recipe v1 unchanged, BPDA, seg-partials, thr-aware,
# weights 1,1,1, craft-encoders all, m30_f15, craft-seed 42.
# Chunked by speaker offset per the section 9 amendment; the chunks tile the 31
# cross-session speakers exactly once.
BASE="--dataset librispeech --subset $SUBSET --cross-session --utt-index 0 \
 --steps 200 --purifier-ckpt $CK --sampler ddim --thresholds $THR \
 --eval-t 50,100,200 --thr-aware --weights 1,1,1 --craft-encoders all \
 --seg-partials --grad-estimator bpda --craft-seed 42 --prereg $PREREG"

craft_chunk() {
  tag=$1; off=$2; n=$3; shift 3
  OUT=$R/main_table_dc_m30_seg_bpda_c${off}.csv
  if skip_guard "$tag" "$OUT"; then say "skip $tag chunk@$off"; return; fi
  say "start $tag chunk offset=$off n=$n"
  python experiments/exp_purifier_aware.py $BASE --spk-offset $off --n-files $n \
    --tag $tag --out $OUT "$@" > $R/dc5_${tag}_c${off}.log 2>&1
  say "done  $tag chunk@$off (exit $?)"
}

# DC5_STAGE=craft|eval|all (default all). `eval` exists because the crafting
# stage is one-shot: once its chunks are on disk, re-entering it meets the
# skip-guard comparing their stored code_sha to the current one, and prereg
# section 7.5 forbids SKIPGUARD_FORCE_SKIP here. Separable stages beat a choice
# between re-crafting 2.3 GPU-h and overriding a guard.
STAGE=${DC5_STAGE:-all}
if [ "$STAGE" = "eval" ]; then say "DC5_STAGE=eval -- skipping crafting"; fi
if [ "$STAGE" != "eval" ]; then
for OFF_N in "0 8" "8 8" "16 8" "24 7"; do
  set -- $OFF_N
  craft_chunk full_dc_m30_seg_bpda    "$1" "$2" --variant full --vocoder phase \
    --margin-db 30 --snr-floor 15
done
for OFF_N in "0 8" "8 8" "16 8" "24 7"; do
  set -- $OFF_N
  craft_chunk fullnsT_dc_m30_seg_bpda "$1" "$2" --variant full --vocoder phase \
    --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
done
fi
if [ "$STAGE" = "craft" ]; then say "DC5_STAGE=craft -- stopping here"; exit 0; fi

# --------------------------------------------------------------- 2. assemble
# The union check that gates everything downstream (prereg section 9).
say "assembling chunks and checking the speaker union"
python experiments/assemble_devclean.py ${DC5_ASSEMBLE_ARGS:-} \
  || { say "FATAL: union check failed"; exit 1; }

# --------------------------------------------- 3. the 12 cells: block (c)
# Mismatched purifier, evaluation only. Thresholds are the TEST-CLEAN foreign
# calibrations -- the whole point of the run is that the operating point was
# not set on these speakers.
TAGS=full_dc_m30_seg_bpda,fullnsT_dc_m30_seg_bpda
for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  [ -f "$CKPT" ] || { say "FATAL: $CKPT absent -- cells for $TAG cannot be run"; exit 1; }
  for T in 1 3; do
    O=$R/dc5_foreign_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing -- refusing any other file"; exit 1; }
    FTHR=$(thr_of "$CAL")
    say "CELL BLOCK: $TAG t=$T, test-clean thresholds $FTHR"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$FTHR" --out "$O" \
      --prereg $PREREG > $R/dc5_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

# ------------------------------------------------- 4. control, NOT a cell
# No purifier. Outside the 12-cell Holm family by construction (prereg
# section 6), reported beside it the way RESULTS 27c is, never inside it.
if [ ! -f $R/dc5_control_nopurifier.csv ]; then
  CTHR=$(thr_of "$R/thresholds_clean_xsess_tc.csv")
  say "CONTROL (not one of the 12 cells): no purifier, thresholds $CTHR"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --thresholds "$CTHR" --out $R/dc5_control_nopurifier.csv \
    --prereg $PREREG > $R/dc5_control.log 2>&1
  say "done (exit $?)"
fi

say "DONE. Decide with: python experiments/analyse_devclean.py"
say "Read the verdict against prereg section 6 -- outcome A, B or C -- and do"
say "not re-run at another budget, timestep, purifier or speaker subset."
