#!/usr/bin/env bash
# ============================================================================
# P0 RERUN — segment-level crafting, both settings, so the fix is ATTRIBUTABLE.
#
# WHAT P0 FIXED
# -------------
# The crafting objective for Resemblyzer was ~0.13 cosine away from the function
# that scores us. We optimised one pooled embedding over the whole clip; the
# evaluator splits into 1.6 s partials, embeds each and averages. Measured on 6
# test-clean speakers (experiments/diag_dvec_proxy.py):
#
#     pooled vs real scoring path   0.882
#     pooled vs partial-averaged    0.872   <- the whole gap
#     partial-averaged vs real      0.977   <- preprocessing is NOT the cause
#
# After the fix: 0.977. Two changes, and they are separate things:
#   (1) FIDELITY  -- embed() now returns the partial-average, the scored quantity.
#   (2) OPTIMISATION -- --seg-partials takes a smooth MAX over per-partial
#       cosines, so the optimiser pushes the WORST partial instead of settling for
#       a good mean. This is AntiFake Sec. C2's weighted segment-based deviation:
#       "the impact of a partial embedding not close to the target is mitigated by
#        other partials that may have attained better optimal points."
#
# WHY BOTH SETTINGS
# -----------------
# The whole point of P0 was that Resemblyzer sat at 35-50 % clone success in the
# DEPLOYMENT setting *while being crafted against*. So deployment is the setting
# that tests the fix. But the STRICT arm must be re-run too, or the strict and
# deployment columns sit on different crafting code and the comparison between
# them -- which is the paper's encoder-overlap measurement -- becomes invalid.
#
# Everything else is held fixed against the runs it replaces: test-clean, the
# same 29 cross-session speakers, purifier v7, cross-session thresholds, loud
# budget, DiffWave at t=3. Only the crafting objective changes.
#
# Compare against:
#   deployment  cloning_tc_m30_deploy.csv   (pooled)  -> cloning_tc_m30_deploy_seg.csv
#   strict      cloning_tc_m30_nodvec.csv   (pooled)  -> cloning_tc_m30_nodvec_seg.csv
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NFILES=29
R=experiments/results
say() { echo "[p0] $(date +%H:%M:%S) $*"; }

THRFILE=$R/thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv
[ -f "$THRFILE" ] || { say "FATAL: $THRFILE missing"; exit 1; }
THR=$(python - <<EOF
from utils.csvio import read_csv
d = read_csv("$THRFILE").set_index("encoder").threshold
print(f"{d['WavLM-SV']:.4f},{d['ECAPA-TDNN']:.4f},{d['Resemblyzer d-vector']:.4f}")
EOF
)
say "thresholds: $THR"

COMMON="--dataset librispeech --subset $SUBSET --cross-session --spk-offset 0 \
 --utt-index 0 --n-files $NFILES --steps 200 --purifier-ckpt $CK --sampler ddim \
 --thresholds $THR --eval-t 50,200 --thr-aware --weights 1,1,1 --seg-partials"

craft() {   # $1 = out csv, $2 = craft-encoders, $3 = tag
  if grep -q "^$3," "$1" 2>/dev/null; then say "skip $3"; return; fi
  say "start $3"
  python experiments/exp_purifier_aware.py $COMMON --craft-encoders "$2" \
    --out "$1" --tag "$3" \
    --variant full --vocoder phase --margin-db 30 --snr-floor 15 \
    --lam-null 10 --null-t 200 > $R/p0_$3.log 2>&1
  say "done  $3 (exit $?)"
}

# ---- A. DEPLOYMENT: all encoders crafted. This is the setting P0 targets.
craft $R/main_table_tc_deploy_seg.csv all fullnsT_m30_seg_all

say "cloning, DEPLOYMENT + segments"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --cross-session --n-files $NFILES --spk-offset 0 \
  --ref-utt 0 --enroll-utt 0 --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK --thresholds $R/thresholds_clean_xsess_tc.csv \
  --variants fullnsT_m30_seg_all --out $R/cloning_tc_m30_deploy_seg.csv \
  > $R/p0_cloning_deploy.log 2>&1
say "done (exit $?)"

# ---- B. STRICT: cloner's encoder held out. Needed so the two columns stay
#         comparable; without it the encoder-overlap measurement is confounded.
craft $R/main_table_tc_seg.csv no-dvec fullnsT_m30_seg_nodvec

say "cloning, STRICT + segments"
python experiments/exp_cloning_protection.py \
  --subset $SUBSET --cross-session --n-files $NFILES --spk-offset 0 \
  --ref-utt 0 --enroll-utt 0 --purifiers none,ours,diffwave --dw-t 3 \
  --ours-ckpt $CK --thresholds $R/thresholds_clean_xsess_tc.csv \
  --variants fullnsT_m30_seg_nodvec --out $R/cloning_tc_m30_nodvec_seg.csv \
  > $R/p0_cloning_strict.log 2>&1
say "done (exit $?)"

say "all done. Read with: python experiments/analyse_p0.py"
