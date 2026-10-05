#!/usr/bin/env bash
# ============================================================================
# FIX S6 -- SEED VARIANCE OF THE SHIPPING RECIPE (council v2, 2026-09-25: "the
# single analysis still owed before publication").
#
# Every crafting run so far used craft()'s default seed 42. The same recipe has
# read 6/0/29 % clone success in one run and 0/0/12 % in another (RESULTS 29e),
# so no cloning number, and no paired delta of 0.01-0.03, has a credible error
# bar. This re-crafts the two EOT arms of the shipping recipe under two more
# seeds (7, 99), giving THREE crafting seeds with the existing seed-42 files.
#
# Estimator: EXACT gradients, because the shipping files (fullnsT_tc_m30_seg)
# were crafted with exact gradients and this measures THEIR spread. RESULTS 30
# shows BPDA is equivalent; using it here would halve the cost but change one
# more thing than the question asks. ~8.5 h crafting + ~2.5 h evaluation.
#
# For each new seed: craft full + fullnsT -> foreign evaluation (4 cells, own
# corrected calibrations) -> no-purifier control -> DiffVC cloning of the
# fullnsT arm (none + De-AntiFake stage 1). Outputs are suffixed _s7 / _s99;
# nothing existing is overwritten. Analysis: per-encoder spread across seeds of
# protection rate, paired mismatch delta, and clone success.
#
# Run:  bash experiments/run_s6_seeds.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1
CK=checkpoints/purifier_v7.pt; SUBSET=test-clean; NFILES=29; R=experiments/results
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[s6] $(date +%H:%M:%S) $*"; }
thr_of() { python experiments/_thr_triple.py "$1"; }
THR=$(thr_of $R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv)

for SEED in 7 99; do
  OUT=$R/main_table_tc_m30_seg_s${SEED}.csv
  BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
   --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
   --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
   --craft-encoders all --seg-partials --grad-estimator exact --craft-seed $SEED --out $OUT"
  run() {
    tag=$1; shift
    if grep -q "^$tag," $OUT 2>/dev/null; then say "skip $tag"; return; fi
    say "start $tag (seed $SEED)"
    python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" > $R/s6_${tag}.log 2>&1
    say "done  $tag (exit $?)"
  }
  run full_tc_m30_seg_s${SEED}    --variant full --vocoder phase --margin-db 30 --snr-floor 15
  run fullnsT_tc_m30_seg_s${SEED} --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
done

TAGS=full_tc_m30_seg_s7,fullnsT_tc_m30_seg_s7,full_tc_m30_seg_s99,fullnsT_tc_m30_seg_s99

for SPEC in "$DW_LIBRI:libri" "$DW_SC09:sc09"; do
  CKPT="${SPEC%%:*}"; TAG="${SPEC##*:}"
  for T in 1 3; do
    O=$R/protect_foreign_s6_${TAG}_t${T}.csv
    [ -f "$O" ] && { say "skip foreign $TAG t=$T"; continue; }
    CAL=$R/thresholds_diffwave_t${T}_${TAG}_xsess_tc.csv
    [ -f "$CAL" ] || { say "FATAL: $CAL missing"; exit 1; }
    say "foreign $TAG t=$T"
    python experiments/eval_saved.py --tags $TAGS --t $T --seeds 2026,7 \
      --n-files $NFILES --spk-offset 0 --utt-index 0 --subset $SUBSET --cross-session \
      --diffwave-ckpt "$CKPT" --thresholds "$(thr_of $CAL)" --out "$O" \
      > $R/s6_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

if [ ! -f $R/control_nopurifier_s6.csv ]; then
  say "control"
  python experiments/eval_saved.py --tags $TAGS --t 0 --seeds 2026 \
    --n-files $NFILES --spk-offset 0 --utt-index 0 --subset $SUBSET --cross-session \
    --thresholds "$(thr_of $R/thresholds_clean_xsess_tc.csv)" --out $R/control_nopurifier_s6.csv \
    > $R/s6_control.log 2>&1
  say "done (exit $?)"
fi

for SEED in 7 99; do
  O=$R/cloning_tc_m30_seg_s${SEED}.csv
  [ -f "$O" ] && { say "skip cloning s$SEED"; continue; }
  say "cloning fullnsT seed $SEED (none + stage 1)"
  python experiments/exp_cloning_protection.py --subset $SUBSET --cross-session \
    --n-files $NFILES --spk-offset 0 --ref-utt 0 --enroll-utt 0 \
    --purifiers none,diffwave --dw-t 3 --thresholds $R/thresholds_clean_xsess_tc.csv \
    --variants fullnsT_tc_m30_seg_s${SEED} --out "$O" > $R/s6_cloning_s${SEED}.log 2>&1
  say "done (exit $?)"
done
say "S6 DONE. Spread across seeds 42/7/99: protection, paired delta, clone success."
