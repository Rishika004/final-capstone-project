#!/usr/bin/env bash
# N7 -- the HELD-OUT contrast at matched budget.
# PREREG: paper/PREREG_N7_heldout_matched.md
#
# Crafts the one arm the project never made -- plain, segment-crafted, with the
# d-vector held out of the ensemble -- and then runs all three held-out seg arms
# through DiffVC in ONE evaluation, so the comparison is within-run (RESULTS 31b:
# 32 % of unprotected cells flip between identical runs, so a cross-run binary
# comparison is not safe).
#
# Flags are byte-identical to run_heldout_seg.sh's full_m30_seg_nodvec except
# --variant plain. One GPU job at a time; the stages are sequential.
set -u
. "$(dirname "$0")/skipguard.sh"
cd "$(dirname "$0")/.." || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_seg_heldout.csv          # same table the other seg arms wrote
PREREG=paper/PREREG_N7_heldout_matched.md
CLONE_OUT=$R/cloning_heldout_seg_n7.csv
say() { echo "[n7] $(date +%H:%M:%S) $*"; }

export TRANSFORMERS_NO_TF=1

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
THR=$(python experiments/_thr_triple.py "$THRFILE")
say "matched thresholds: $THR"

COMMON="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,200 --thr-aware --weights 1,1,1 --seg-partials \
 --craft-encoders no-dvec --prereg $PREREG"

TAG=plain_m30_seg_nodvec
if skip_guard "$TAG" "$OUT"; then
  say "skip craft $TAG (already present)"
else
  say "CRAFT $TAG -- the arm that has never existed"
  python experiments/exp_purifier_aware.py $COMMON \
    --variant plain --margin-db 30 --snr-floor 15 \
    --out "$OUT" --tag "$TAG" > $R/n7_craft.log 2>&1
  say "craft done (exit $?)"
fi

n=$(ls $R/purifier_aware/*_${TAG}.wav 2>/dev/null | wc -l)
say "$n/$NFILES wavs on disk for $TAG"
[ "$n" -lt "$NFILES" ] && { say "FATAL: crafting incomplete, not evaluating"; exit 1; }

say "BUDGET CHECK (prereg 4) -- must be within 0.5 dB and not significant"
python experiments/analyse_n1_budget.py --arms \
  "plain_m30_seg_nodvec,full_m30_seg_nodvec,fullnsT_m30_seg_nodvec" \
  --pair "plain_m30_seg_nodvec:fullnsT_m30_seg_nodvec:HELD-OUT-SEG" \
  --out $R/n7_spend.csv > $R/n7_budget.txt 2>&1
sed -n '/Are the arms at equal spend/,/^$/p' $R/n7_budget.txt

if [ -f "$CLONE_OUT" ]; then
  say "skip cloning ($CLONE_OUT present)"
else
  say "CLONING -- three held-out seg arms, one run, three channels"
  python experiments/exp_cloning_protection.py \
    --subset $SUBSET --cross-session --n-files $NFILES --spk-offset 0 \
    --ref-utt 0 --enroll-utt 0 \
    --purifiers none,ours,diffwave --dw-t 3 --ours-ckpt $CK \
    --thresholds $R/thresholds_clean_xsess_tc.csv \
    --variants plain_m30_seg_nodvec,full_m30_seg_nodvec,fullnsT_m30_seg_nodvec \
    --clone-seed 2026 \
    --out "$CLONE_OUT" --prereg $PREREG > $R/n7_cloning.log 2>&1
  say "cloning done (exit $?)"
fi

say "ANALYSIS (prereg 5: read the saturation check before the branch)"
python experiments/analyse_n5.py --in "$CLONE_OUT" \
  --ship fullnsT_m30_seg_nodvec --plain plain_m30_seg_nodvec \
  --budget-csv $R/n7_spend.csv | tee $R/n7_analysis.txt

say "N7 DONE. Read the budget gate first; a failed match voids the verdict."
