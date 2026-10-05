#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"   # W0: refuse to skip rows written by other code
# ============================================================================
# ROADMAP P1.3 — the LOUD budget with the cloner's encoder HELD OUT.
#
# THE QUESTION
# ------------
# Our honest cloning number is 38-60 % clone success, which is weak. Two
# explanations, and they call for completely different work:
#
#   (a) BUDGET. The loud budget m30_f15 already gives 1-5 % clone success
#       (cloning_protection.csv). But those runs crafted against all three
#       encoders, including the one DiffVC itself uses, so they are white-box.
#       If the loud budget stays near zero with that encoder held out, then the
#       38-60 % was an artefact of the TIGHT budget and the headline is fine.
#
#   (b) TRANSFER. If the loud budget degrades the same way the tight one did,
#       then encoder mismatch is the real barrier, and the fix is a bigger
#       crafting ensemble (P1.1), not a louder perturbation.
#
# Either answer is worth having, and the experiment costs no new code. This is
# why it runs before P1.1: it decides how much P1.1 is worth.
#
# This run also doubles as the test-clean re-craft, so nothing is done twice.
#
# PREREQUISITE: run_testclean_pipeline.sh must have passed both gates.
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc.csv
say() { echo "[p13] $(date +%H:%M:%S) $*"; }

THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
if [ ! -f "$THRFILE" ]; then
  say "FATAL: $THRFILE missing. Run run_testclean_pipeline.sh first."; exit 1
fi
THR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$THRFILE").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
say "thresholds (v7/ddim t=50, cross-session, $SUBSET): $THR"

# --spk-offset 0 is stated EXPLICITLY. Relying on the default is what broke the
# first attempt: the default was 20, chosen when the purifier trained on dev-clean
# speakers 0-19, and it quietly reduced 29 speakers to 9 while the cloning step
# below still asked for all 29.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 --utt-index 0 \
 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,100,200 --thr-aware --weights 1,1,1 \
 --craft-encoders no-dvec --out $OUT"

run() {
  tag=$1; shift
  if skip_guard "$tag" "$OUT"; then
    say "skip $tag (already present)"; return
  fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/tc_${tag}.log 2>&1
  say "done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' $R/tc_${tag}.log | grep "t= 50\|t=200\|quality"
}

# LOUD budget first -- it answers the question this script exists for.
run full_m30_f15_nodvec    --variant full --vocoder phase --margin-db 30 --snr-floor 15
run fullnsT_m30_f15_nodvec --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run plain_m30_f15_nodvec   --variant plain --margin-db 30 --snr-floor 15

say "loud budget crafted -- starting cloning evaluation"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --cross-session --n-files $NFILES --spk-offset 0 \
  --ref-utt 0 --enroll-utt 0 \
  --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK \
  --thresholds $R/thresholds_clean_xsess_tc.csv \
  --variants plain_m30_f15_nodvec,full_m30_f15_nodvec,fullnsT_m30_f15_nodvec \
  --out $R/cloning_tc_m30_nodvec.csv \
  > $R/tc_cloning_m30.log 2>&1
say "cloning done (exit $?)"
grep -A12 "clone success rate" $R/tc_cloning_m30.log | tail -20

say "all done -- compare against cloning_protection_f1.csv (tight budget, dev-clean)"
