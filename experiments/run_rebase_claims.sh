#!/usr/bin/env bash
# ============================================================================
# REBASE THE VERIFICATION CLAIMS onto the corrected evaluation base.
#
# THE PROBLEM THIS FIXES
# ----------------------
# The paper's four strongest claims -- survival, transfer to purifiers we did not
# train, the mismatch trade, perceptual cost -- are all still measured on:
#     dev-clean + purifier v3 + same-session thresholds + 20 speakers
# while every cloning result has moved to:
#     test-clean + purifier v7 + cross-session thresholds + 29 speakers
#
# Individually both are fine. Together they are incoherent: a reader comparing
# the verification section against the cloning section finds two different
# evaluation protocols, and the corrected one is strictly harsher. Nothing here
# is expected to overturn a claim -- the paired statistics are threshold-invariant
# -- but the numbers will move and the paper must quote one base.
#
# WHAT CHANGES, AND WHAT MUST NOT
#   corpus      dev-clean -> test-clean      (what AntiFake and De-AntiFake use)
#   purifier    v3        -> v7              (trained on train-clean-100, disjoint)
#   thresholds  same-session -> cross-session (fix S1)
#   speakers    20        -> 29              (all that support cross-session)
#   selection   ad-hoc    -> shared selector  (crafting/cloning/eval all agree)
#
# EXPECT ABSOLUTE NUMBERS TO GET WORSE. Cross-session thresholds are lower, so
# every encoder accepts more readily. What should NOT move is the paired
# comparisons: fix S2 changed absolute rates while leaving p-values identical to
# four decimals, because the threshold cancels in a difference. If a PAIRED
# result moves materially here, that is a finding to investigate, not absorb.
#
# Run:  bash experiments/run_rebase_claims.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
DW=checkpoints/foreign/diffwave_libri_deantifake.pkl
say() { echo "[rebase] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }

# ---------------------------------------------------------------- 1. survival
# Claim: the perturbation survives purification, and the purifier has a blind end
# and a vicious end. Pure geometry -- no encoders, no thresholds involved.
if [ ! -f $R/survival_tc_v7.csv ]; then
  say "survival (alpha), 29 speakers, v7/ddim"
  python experiments/exp_nullspace_survival.py \
    --n-files $NFILES --spk-offset 0 --utt-index 0 \
    --subset $SUBSET --cross-session \
    --t 50,100,200,300 --null-t 200 --seeds 2026,7 --sampler ddim \
    --purifier-ckpt $CK --out $R/survival_tc_v7.csv \
    > $R/rebase_survival.log 2>&1
  say "done (exit $?)"
  grep -i "MEAN" $R/rebase_survival.log | head -4
else
  say "skip survival (already present)"
fi

# ---------------------------------------------------------------- 2. transfer
# Claim: the subspace transfers to purifiers we did NOT train. The strongest
# result in the paper, so it matters most that it sits on the final base.
for FOREIGN in "$DW:libri" "checkpoints/foreign/diffwave_sc09_philsyn.pkl:sc09"; do
  CKPT="${FOREIGN%%:*}"; TAG="${FOREIGN##*:}"
  [ -f "$CKPT" ] || { say "skip $TAG (checkpoint absent)"; continue; }
  for T in 1 3; do
    OUT=$R/foreign_tc_${TAG}_t${T}.csv
    if [ -f "$OUT" ]; then say "skip $TAG t=$T"; continue; fi
    say "transfer: $TAG t=$T"
    python experiments/exp_foreign_purifier.py \
      --n-files $NFILES --spk-offset 0 --utt-index 0 \
      --subset $SUBSET --cross-session \
      --foreign-ckpt "$CKPT" --foreign-t $T \
      --purifier-ckpt $CK --sampler ddim \
      --out "$OUT" > $R/rebase_foreign_${TAG}_t${T}.log 2>&1
    say "done (exit $?)"
  done
done

say "survival + transfer rebased."
say "STILL TO DO on this base: the mismatch trade (4 arms x 29 speakers at the"
say "m20 budget, plus foreign-purifier evaluation and the no-purifier control)."
say "That is the expensive one and is queued separately."
