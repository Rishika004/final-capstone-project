#!/usr/bin/env bash
. "$(dirname "$0")/skipguard.sh"   # W0: refuse to skip rows written by other code
# ============================================================================
# THE COMPARABLE ROW — our numbers on De-AntiFake's exact published speaker set,
# reported under BOTH threshold protocols.
#
# WHY TWO ROWS AND NOT ONE
# ------------------------
# De-AntiFake evaluates on 25 named test-clean speakers. Only 18 of those have a
# second chapter, so cross-session thresholds (fix S1) cannot cover the other
# seven. That forces a choice, and taking either option alone loses something:
#
#   their 25, same-session   directly comparable, but uses the calibration
#                            protocol we documented as broken (0.03 % EER)
#   our 18, cross-session    honest, but a smaller set than theirs and no longer
#                            a like-for-like comparison
#
# So we report BOTH. The gap between the rows is not an inconvenience, it is a
# measurement: how much a same-session protocol flatters this literature, on the
# exact speakers a published paper used. Nobody else reports it, and it costs us
# almost nothing because the crafting is shared -- only the scoring differs.
#
# COMPARABILITY NOTES TO CARRY INTO THE PAPER
#   * Their SVA (speaker verification accuracy) is the SAME quantity as our clone
#     success rate: the fraction of clones a verifier accepts. No translation.
#   * They use 5 sentences per speaker; we use 1. Sample sizes do not match even
#     where the speakers do.
#   * Their xSVA threshold is 0.25 on a SpeechBrain system they label "x-vector"
#     while citing the ECAPA paper. We report ECAPA at 0.25 alongside our
#     EER-calibrated figure so both readings are available.
#
# ENSEMBLE SETTING: --craft-encoders all (DEPLOYMENT), corrected 2026-09-19.
# This row is a COMPARISON against De-AntiFake, so it must use the setting they
# use. They reproduce AntiFake with all four of its encoders in the ensemble,
# GE2E included. Running our row in the strict held-out setting would compare our
# hardest condition against their standard one and understate us for no reason.
# The strict number is reported separately (RESULTS section 22).
#
# PREREQUISITE: run_testclean_pipeline.sh passed; purifier v7 exists.
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
R=experiments/results
OUT=$R/main_table_daf.csv
say() { echo "[daf] $(date +%H:%M:%S) $*"; }

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
say "cross-session thresholds: $THR"

# ---------------------------------------------------------------- 1. calibrate
# Their 25 cannot use the cross-session calibration (7 speakers lack a second
# chapter), so the comparable row needs a SAME-SESSION calibration on exactly
# those 25. This file is produced ONLY for that row and is labelled as such.
if [ ! -f $R/thresholds_clean_daf25_samesession.csv ]; then
  say "calibrating clean channel, same-session, De-AntiFake's 25 (for the comparable row)"
  python experiments/calibrate_thresholds.py --channel clean --subset $SUBSET \
    --speaker-set deantifake --n-spk 25 --n-utt 4 --tag daf25_samesession \
    > $R/calib_daf25_samesession.log 2>&1
  grep "EER" $R/calib_daf25_samesession.log
fi

# ---------------------------------------------------------------- 2. craft
# Craft on the UNION: all 25. The 18 cross-session speakers are a subset, so one
# crafting pass serves both rows. --speaker-set restricts to their list; without
# --cross-session it takes all 25.
BASE="--dataset librispeech --subset $SUBSET --speaker-set deantifake \
 --spk-offset 0 --utt-index 0 --n-files 25 --steps 200 --purifier-ckpt $CK \
 --sampler ddim --thresholds $THR --eval-t 50,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --out $OUT"

run() {
  tag=$1; shift
  if skip_guard "$tag" "$OUT"; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > $R/daf_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run fullnsT_daf --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200

# ---------------------------------------------------------------- 3. two rows
say "ROW A -- their 25 speakers, same-session thresholds (comparable to their table)"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --speaker-set deantifake --n-files 25 --spk-offset 0 \
  --ref-utt 0 --enroll-utt 1 --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK --thresholds $R/thresholds_clean_daf25_samesession.csv \
  --variants fullnsT_daf --out $R/cloning_daf25_samesession.csv \
  > $R/daf_cloning_rowA.log 2>&1
say "row A done (exit $?)"

say "ROW B -- the 18 that support cross-session, honest thresholds"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --speaker-set deantifake --cross-session --n-files 18 \
  --spk-offset 0 --ref-utt 0 --enroll-utt 0 --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK --thresholds $R/thresholds_clean_xsess_tc.csv \
  --variants fullnsT_daf --out $R/cloning_daf18_xsession.csv \
  > $R/daf_cloning_rowB.log 2>&1
say "row B done (exit $?)"

say "both rows written. Compare:"
say "  A  $R/cloning_daf25_samesession.csv   (their protocol, their speakers)"
say "  B  $R/cloning_daf18_xsession.csv      (our protocol, the overlap)"
say "The A-vs-B gap is what the calibration protocol is worth."
