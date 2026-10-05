#!/usr/bin/env bash
# ============================================================================
# THE DEPLOYMENT SETTING — craft against every encoder a real defender can get,
# including the one DiffVC uses. This is the number that answers "does this stop
# cloning?", and it is reported BESIDE the held-out number, never instead of it.
#
# WHY THIS IS LEGITIMATE, AND WHY IT IS NOT A RETREAT FROM FIX F1
# ---------------------------------------------------------------
# F1's finding was NOT "including Resemblyzer is cheating". It was "calling a
# result a TRANSFER result while including the target's own encoder is
# mislabelling". The number was never the problem; the label was.
#
# What the field actually does, verified against the sources this session:
#   * AntiFake (CCS 2023) crafts against FOUR encoders, GE2E among them, and
#     reports results on SV2TTS -- which USES GE2E. It does not hold it out.
#     Its "black-box" claim is about the SYNTHESIZER, not the encoder.
#   * De-AntiFake reproduces AntiFake with all four encoders in the ensemble.
#
# And it matches the real threat model. Resemblyzer is a free pip install. A
# defender protecting their own voice would obviously include it. Holding it out
# models an attacker whose identity encoder is SECRET -- a strictly stronger
# adversary than any published paper assumes.
#
# So we report two settings and name them honestly:
#
#   DEPLOYMENT  all public encoders in the ensemble, including the cloner's.
#               Black-box w.r.t. the cloning SYSTEM. Comparable to AntiFake.
#               -> this is the "does it stop cloning" number.
#
#   STRICT      the cloner's encoder held out. Black-box w.r.t. the ENCODER too.
#               Stronger than anything published. Already run: RESULTS §22.
#               -> this is the honest upper bound under a secret encoder.
#
# The GAP between them is itself a contribution: it measures how much any
# protection result in this literature depends on encoder overlap, a variable
# almost nobody reports. We can report it because we ran both.
#
# PREREQUISITE: run_testclean_pipeline.sh passed; purifier v7 exists.
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
OUT=$R/main_table_tc_deploy.csv
say() { echo "[deploy] $(date +%H:%M:%S) $*"; }

THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
THR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$THRFILE").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
say "thresholds: $THR"

# --craft-encoders all  <- the ONLY difference from run_p13_loud.sh.
# Same corpus, same speakers, same purifier, same thresholds, same budget, same
# arms. So the deployment-vs-strict gap is attributable to one variable.
BASE="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 --utt-index 0 \
 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,200 --thr-aware --weights 1,1,1 \
 --craft-encoders all --out $OUT"

run() {
  tag=$1; shift
  if grep -q "^$tag," $OUT 2>/dev/null; then say "skip $tag"; return; fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" > $R/dep_${tag}.log 2>&1
  say "done  $tag (exit $?)"
}
run fullnsT_m30_f15_all --variant full --vocoder phase --margin-db 30 --snr-floor 15 --lam-null 10 --null-t 200
run plain_m30_f15_all   --variant plain --margin-db 30 --snr-floor 15

say "crafting done -- cloning evaluation (deployment setting)"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --cross-session --n-files $NFILES --spk-offset 0 \
  --ref-utt 0 --enroll-utt 0 --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK --thresholds $R/thresholds_clean_xsess_tc.csv \
  --variants plain_m30_f15_all,fullnsT_m30_f15_all \
  --out $R/cloning_tc_m30_deploy.csv \
  > $R/dep_cloning.log 2>&1
say "cloning done (exit $?)"

say "compare: cloning_tc_m30_deploy.csv (DEPLOYMENT) vs cloning_tc_m30_nodvec.csv (STRICT)"
