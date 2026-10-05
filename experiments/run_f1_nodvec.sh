#!/usr/bin/env bash
# F1 -- drop Resemblyzer from CRAFTING, keep it in EVALUATION.
#
# WHY. DiffVC's speaker conditioner is a GE2E checkpoint byte-identical to the
# Resemblyzer encoder in our crafting ensemble (md5 a2e2a0bf4cc3ac59053d3fe4d67b0c4b).
# Every cloning number we have is therefore a matched-encoder number: the
# perturbation was optimised against the exact network the cloner reads identity
# with. AntiFake (CCS 2023, Table 4) measured the size of that gap: single-encoder
# crafting protects the synthesizer built on that encoder at 99.6-100 % and the
# other four at 0-20.6 %. Their GE2E row is our case exactly -- 100 % on the
# synthesizer that uses GE2E, 1.2-20.6 % on those that do not. So this is not a
# footnote, it decides whether the cloning claim is about deployment or about a
# coincidence of checkpoints.
#
# WHAT IS CONTROLLED. Everything except the crafting ensemble:
#   same corpus  dev-clean, pinned with --subset (EVAL_SUBSET moved to test-clean
#                on 2026-09-17; re-crafting under the new default would confound
#                the ablation with a change of speakers)
#   same speakers        --spk-offset 20 --utt-index 3 --n-files 20
#   same purifier        checkpoints/purifier_v3.pt, ddim
#   same thresholds      thresholds_diffpure_gl_t50_ddim_v3.csv
#   same budgets/steps   m20_f20, 200 steps
# The only difference from run_queue7/8/10 is --craft-encoders no-dvec.
#
# READING THE RESULT. Compare arm-for-arm against the published tags:
#   ASV      main_table_v3.csv          vs  main_table_f1.csv
#   cloning  cloning_protection_m20.csv vs  cloning_protection_f1.csv
# The d-vector column of the new run is a HELD-OUT encoder. If protection there
# collapses, the cloning claim was white-box and must be restated. If it survives,
# the claim transfers and gets stronger, not weaker.
cd "$(dirname "$0")/.."
CK=checkpoints/purifier_v3.pt
THR=0.897,0.3457,0.7351          # WavLM-SV, ECAPA, d-vector (v3/ddim t=50)
say() { echo "[f1] $(date +%H:%M:%S) $*"; }

BASE="--dataset librispeech --subset dev-clean --spk-offset 20 --utt-index 3 \
 --n-files 20 --steps 200 --purifier-ckpt $CK --sampler ddim --thresholds $THR \
 --eval-t 50,100,200 --thr-aware --weights 1,1,1 --craft-encoders no-dvec \
 --out experiments/results/main_table_f1.csv"

run() {
  tag=$1; shift
  # `variant` is the FIRST csv column, so a leading comma never matches. Anchor
  # at line start instead. Getting this wrong re-ran a completed arm and would
  # have appended a duplicate set of rows to the results file.
  if grep -q "^$tag," experiments/results/main_table_f1.csv 2>/dev/null; then
    say "skip $tag (already in main_table_f1.csv)"; return
  fi
  say "start $tag"
  python experiments/exp_purifier_aware.py $BASE --tag $tag "$@" \
    > experiments/results/f1_${tag}.log 2>&1
  say "done  $tag (exit $?)"
  sed -n '/MEAN PER ENCODER/,$p' experiments/results/f1_${tag}.log \
    | grep "t= 50\|t=100\|t=200\|quality"
}

run plain_m20_f20_nodvec    --variant plain --margin-db 20 --snr-floor 20
run nsT_m20_f20_nodvec      --variant plain --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200
run full_m20_f20_nodvec     --variant full --vocoder phase --margin-db 20 --snr-floor 20
run fullnsT_m20_f20_nodvec  --variant full --vocoder phase --margin-db 20 --snr-floor 20 --lam-null 10 --null-t 200

say "crafting done -- starting cloning evaluation"
python experiments/exp_cloning_protection.py \
  --subset dev-clean --n-files 20 --spk-offset 20 --ref-utt 3 --enroll-utt 0 \
  --purifiers none,ours,diffwave --dw-t 1 \
  --variants plain_m20_f20_nodvec,nsT_m20_f20_nodvec,full_m20_f20_nodvec,fullnsT_m20_f20_nodvec \
  --out experiments/results/cloning_protection_f1.csv \
  > experiments/results/f1_cloning.log 2>&1
say "cloning done (exit $?)"
say "all done"
