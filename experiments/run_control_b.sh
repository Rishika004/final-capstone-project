#!/usr/bin/env bash
# ============================================================================
# CONTROL B -- does the word "speaker" in C1's title do any work?
#
# PREREG_council5_controls.md section B, written 2026-10-07 BEFORE any of this
# was run. Two reviewers found the gap independently.
#
# THE GAP
# -------
# Every existing C1 control -- band-matched, pca_span, white noise -- tests a
# SPECTRAL alternative. None tests the TASK alternative:
#
#   "a speech denoiser removes the directions a differentiable speech network
#    is most sensitive to"   <- a generic fact about neural Jacobians
#   "...that a SPEAKER encoder reads"   <- the paper's claim
#
# They are currently indistinguishable and the title rests on the difference.
#
# WHAT IS RUN
# -----------
# alpha for the top gradient direction of an UNTRAINED CAM++ -- identical
# architecture, identical front end, identical gradient path, no learned
# weights -- on the same purifier, the same 29 speakers, the same --rms and
# the same seeds, through the same survival() path. Three random inits, since
# one init is one draw; the MEAN over them is the pre-declared primary and
# each is also reported.
#
# The premise (that an untrained net carries no identity) is VERIFIED in-run
# rather than asserted. Measured on 12 speakers, CPU, before this script was
# written:
#     seed 0   speaker separation +0.0197
#     seed 1   speaker separation +0.0061
# against the trained CAM++'s +0.614. The pre-registration quoted +0.038 from
# an uncommitted check; this is the committed version and agrees.
#
# READING, fixed in advance by the pre-registration (section B.3). Let
# alpha_u be the untrained encoder's mean alpha at t = 50, against trained
# encoders at 0.388-0.631 and `random` at 0.714:
#
#   B-1  alpha_u <= ~0.63  (inside the trained band)
#        -> C1's TITLE IS WRONG. The effect is about gradients of deep speech
#           networks through a denoiser, not about speaker identity.
#           "Speaker" comes out of the title, C1 is demoted, and the paper
#           leads with the measurement contribution. FRAME_v7 section 2 holds
#           the fallback title ready.
#   B-2  alpha_u >= ~0.70  (near `random`)
#        -> C1 SURVIVES AND IS STRENGTHENED. An untrained network of the same
#           architecture is not purifier-hostile, so the effect tracks LEARNED
#           identity rather than architecture or gradient roughness. This is
#           the control that earns the word "speaker"; it goes in Fig. 1.
#   B-3  0.63 < alpha_u < 0.70
#        -> report the number and weaken to "learned speaker encoders are more
#           purifier-hostile than an untrained network of the same
#           architecture, by X", with no mechanism language.
#
# BINDING (section B.4): this control may NOT be re-run at other settings to
# obtain a different verdict. And an untrained network is a weak proxy for "no
# identity" -- it also has untrained FEATURES -- so if B-1 fires the honest
# statement is the disjunction "identity or architecture", not "architecture".
#
# A FURTHER CAVEAT, found while building this and recorded before the result:
# the untrained network's embeddings are nearly DEGENERATE (same-speaker
# cosine 0.995, different-speaker 0.975-0.990), i.e. it maps everything to
# almost one point. Its gradient is therefore the direction that breaks a
# near-constant function, which is not obviously the same KIND of object as a
# trained encoder's identity direction. This weakens B-2 as evidence (an
# unusual direction may survive differently for reasons unrelated to identity)
# and must be stated whichever branch fires.
#
# COST: ~1.5 GPU-h -- the same per-speaker cost as the alpha runs, with three
# extra directions plus their band-matched controls.
#
# Run:  bash experiments/run_control_b.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

R=experiments/results
OUT=$R/encoder_subspace_controlb.csv
PREREG=paper/PREREG_council5_controls.md
CK=checkpoints/purifier_v7.pt
say() { echo "[ctrlB] $(date +%H:%M:%S) $*"; }

[ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
[ -f "$PREREG" ] || { say "FATAL: $PREREG missing"; exit 1; }
if [ -f "$OUT" ]; then
  say "REFUSING: $OUT exists. B.4 forbids re-running this control to obtain a"
  say "different verdict. Move or delete it deliberately if you must re-run."
  exit 1
fi

say "alpha for 3 untrained CAM++ inits + the 5 trained encoders, 29 speakers"
python experiments/exp_encoder_subspace.py \
  --n-files 29 --subset test-clean --cross-session \
  --t 50,200 --seeds 2026,7 --rms 0.2 \
  --purifier-ckpt $CK --sampler ddim \
  --with-extra --with-untrained --untrained-seeds 0,1,2 \
  --pca-subset dev-clean \
  --out "$OUT" --prereg "$PREREG" \
  > $R/control_b.log 2>&1
say "done (exit $?)"
say "Read with: python experiments/analyse_control_b.py"
