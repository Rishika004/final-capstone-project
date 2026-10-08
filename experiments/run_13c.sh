#!/usr/bin/env bash
# ============================================================================
# COUNCIL #13c -- alpha through DiffWave. The last structural hole in C1.
#
# C1's alpha is measured on OUR mel purifier (v7). The phenomenon it explains
# -- the constraint's encoder-dependent effect and the d-vector reversal --
# lives on DiffWave at t = 1 and t = 3, a WAVEFORM model. Explanation and
# explanandum on different purifiers, in different representations.
#
# PLAN_V3 W7, council v4 E.7 (unanimous), FRAME_v6 C1's one remaining
# [PENDING], and R1's "a reviewer sees this in thirty seconds".
#
# WHAT IS RUN
# -----------
# exp_encoder_subspace_wav.py: the same survival() arithmetic as RESULTS 32,
# with waveform directions and a DiffWave purifier. Both checkpoints the paper
# reports, at both evaluation timesteps, 29 speakers, 5 encoders plus a
# band-matched control each plus isotropic random.
#
# THE READING, FIXED BEFORE THE RUN
# ---------------------------------
# DiffWave at t = 1-3 is close to identity-plus-noise, so these alphas will sit
# much nearer 1 than the mel-purifier alphas. THE LEVELS ARE NOT COMPARABLE
# ACROSS PURIFIERS and no row may be quoted as if they were. What is comparable
# is the CONTRAST WITHIN this purifier:
#
#   PASS  every encoder's direction sits below its OWN band-matched control,
#         on both checkpoints, at both timesteps (10 cells per checkpoint)
#         -> C1's level claim is purifier-general, not an artefact of the mel
#            representation or of our own denoiser. The 32/37/43 result and the
#            C2 phenomenon finally sit on the SAME purifier. C1's last PENDING
#            closes.
#   MIXED some cells below, some not
#         -> report the pattern per cell. C1's claim becomes "on our mel
#            purifier, and on DiffWave for the following encoders", which is
#            weaker but honest, and the figure gains a panel rather than a
#            footnote.
#   FAIL  encoder directions are NOT below their band-matched controls on
#         DiffWave
#         -> C1 is a statement about OUR purifier, not about speech purifiers.
#            That is a real finding and it is reported as one: the mechanism
#            explaining the DiffWave phenomenon would then not hold on
#            DiffWave, which weakens C1-as-explanation substantially and must
#            be said in the abstract, not the limitations.
#
# No cell may be dropped after the fact, and the two checkpoints are read
# together: picking the one that agrees is the 25 failure mode.
#
# COST: ~2.5-3 GPU-h. DiffWave at t<=3 is a 1-3 step reverse loop, so the cost
# is dominated by the per-encoder gradients, not the purifier.
#
# Run:  bash experiments/run_13c.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

R=experiments/results
OUT=$R/encoder_subspace_wav.csv
DW_LIBRI=checkpoints/foreign/diffwave_libri_deantifake.pkl
DW_SC09=checkpoints/foreign/diffwave_sc09_philsyn.pkl
say() { echo "[13c] $(date +%H:%M:%S) $*"; }

for CK in "$DW_LIBRI" "$DW_SC09"; do
  [ -f "$CK" ] || { say "FATAL: $CK missing"; exit 1; }
done

if [ -f "$OUT" ]; then
  say "NOTE: $OUT exists; the script appends. Move it aside for a clean run."
fi

for SPEC in "$DW_LIBRI:DiffWave-LibriSpeech:libri" \
            "$DW_SC09:DiffWave-SC09:sc09"; do
  CK="${SPEC%%:*}"; REST="${SPEC#*:}"; NAME="${REST%%:*}"; TAG="${REST##*:}"
  say "alpha through $NAME, t=1,3, 29 speakers, 5 encoders + controls"
  python experiments/exp_encoder_subspace_wav.py \
    --n-files 29 --subset test-clean --cross-session --utt-index 0 \
    --diffwave-ckpt "$CK" --purifier-name "$NAME" \
    --t 1,3 --seeds 2026,7 --rel-rms 0.05 --with-extra \
    --out "$OUT" > $R/13c_${TAG}.log 2>&1
  say "done $NAME (exit $?)"
done

say "DONE. Read with: python experiments/analyse_13c.py"
