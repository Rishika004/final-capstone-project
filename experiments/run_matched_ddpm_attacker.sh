#!/usr/bin/env bash
# ============================================================================
# THE ATTACK THE SECURITY REVIEWER SAID THEY WOULD RUN (council review, 2026-09-22).
#
# We craft with EOT through a DETERMINISTIC DDIM reverse process (differentiable,
# checkpointed). An attacker is not obliged to purify the same way: the same
# purifier weights can be run as a STOCHASTIC DDPM sampler, several seeds. If our
# gradients only "know" the deterministic path, protection should drop here.
# Prediction on record from the reviewer: WavLM/ECAPA clone success climbs from
# 7-13 % toward 40-86 %.
#
# This is EVALUATION ONLY of the already-crafted shipping-recipe files
# (full_tc_m30_seg, fullnsT_tc_m30_seg): no crafting, ~1 h.
#   step 0  calibrate our purifier v7 as a DDPM channel at t=50, cross-session,
#           test-clean (fix S2: every channel scored at its own calibration)
#   step 1  push both arms through v7/DDPM t=50, FIVE seeds, score at that calibration
# Compare against the DDIM row already in main_table_tc_m30_seg.csv (eval_t=50).
#
# Run:  bash experiments/run_matched_ddpm_attacker.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1
CK=checkpoints/purifier_v7.pt; SUBSET=test-clean; NFILES=29; R=experiments/results
say() { echo "[ddpm] $(date +%H:%M:%S) $*"; }
[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }

CAL=$R/thresholds_diffpure_gl_t50_v7_xsess_tc.csv      # no _ddim_ in the name = DDPM
if [ ! -f "$CAL" ]; then
  say "CALIBRATE v7 as DDPM channel, t=50, cross-session, $SUBSET"
  python experiments/calibrate_thresholds.py --channel diffpure_gl --sampler ddpm \
    --t-diffuse 50 --purifier-ckpt $CK --subset $SUBSET --cross-session \
    --n-spk $NFILES --n-utt 4 --tag v7_xsess_tc > $R/ddpm_calib_t50.log 2>&1
  say "done (exit $?)"
fi
[ -f "$CAL" ] || { say "FATAL: calibration did not produce $CAL"; exit 1; }
THR=$(python experiments/_thr_triple.py "$CAL")
say "DDPM t=50 thresholds: $THR"

OUT=$R/protect_matched_ddpm_seg_t50.csv
if [ ! -f "$OUT" ]; then
  say "EVAL: shipping-recipe arms through v7/DDPM t=50, 5 seeds"
  python experiments/eval_saved.py --tags full_tc_m30_seg,fullnsT_tc_m30_seg \
    --t 50 --seeds 2026,7,99,11,42 --sampler ddpm --purifier-ckpt $CK \
    --n-files $NFILES --spk-offset 0 --utt-index 0 --subset $SUBSET --cross-session \
    --thresholds "$THR" --out "$OUT" > $R/ddpm_eval_t50.log 2>&1
  say "done (exit $?)"
fi
say "DONE. Compare clone_fails_after / fails_all_seeds against main_table_tc_m30_seg.csv eval_t=50 (DDIM)."
