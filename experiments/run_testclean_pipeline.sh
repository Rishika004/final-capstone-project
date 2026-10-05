#!/usr/bin/env bash
# ============================================================================
# ROADMAP ITEM 1 — move the whole evaluation onto test-clean with an honest
# purifier and honest thresholds. Fixes S1b (corpus) and S1 (cross-session).
#
# WHY THIS RUNS BEFORE ANYTHING ELSE
# ----------------------------------
# Every remaining experiment produces numbers that go in the paper. All of them
# must sit on the final corpus, the final purifier and the final thresholds, or
# they get run twice. This script establishes that base.
#
# WHAT CHANGES FROM THE PUBLISHED RUN
#   corpus      dev-clean          -> test-clean   (what AntiFake and De-AntiFake use)
#   purifier    trained dev-clean  -> trained train-clean-100  (disjoint from eval)
#   thresholds  same-session       -> cross-session (enrolment from another chapter)
#   speakers    20 (half the split) -> 29 (all that support cross-session)
#
# Expect every absolute number to get WORSE. Same-session calibration and a
# purifier that saw the evaluation split were both flattering us. The paired
# comparisons, which are what the contribution rests on, should be unaffected:
# the threshold cancels in a difference.
#
# GATE. The purifier must clear denoising gain >= 0.15 before anything is
# measured against it. Purifier v2 was effectively an identity map (gain 0.06)
# and everything measured against it had to be discarded. That is the v2 lesson
# and it is enforced here rather than remembered.
#
# Run:  bash experiments/run_testclean_pipeline.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

CK=checkpoints/purifier_v7.pt
SUBSET=test-clean
NSPK=29
R=experiments/results
say() { echo "[tc] $(date +%H:%M:%S) $*"; }

# ---------------------------------------------------------------- gate
if [ ! -f "$CK" ]; then
  say "FATAL: $CK not found. Training has not finished."; exit 1
fi
GAIN=$(python -c "import torch;print(torch.load('$CK',map_location='cpu').get('val_denoise_gain',-9))" 2>/dev/null || echo -9)
say "purifier v7 denoising gain = $GAIN  (gate: >= 0.15)"
if ! python -c "import sys;sys.exit(0 if float('$GAIN')>=0.15 else 1)"; then
  say "FATAL: v7 failed the denoising gate. Do NOT measure anything against it."
  say "Retrain with more steps or a higher lr before continuing."
  exit 1
fi

# ---------------------------------------------------------------- 1. thresholds
# Cross-session, on the corpus we will actually evaluate on, through EVERY
# channel an attacker might use. Reusing one channel's threshold for another is
# fix S2 and is not allowed.
say "calibrating CLEAN channel (cross-session, $SUBSET)"
python experiments/calibrate_thresholds.py --channel clean --subset $SUBSET \
  --cross-session --n-spk $NSPK --n-utt 4 --tag xsess_tc \
  > $R/calib_clean_xsess_tc.log 2>&1
grep "EER" $R/calib_clean_xsess_tc.log

for T in 50 100 200; do
  say "calibrating ours/DDIM t=$T (cross-session, $SUBSET)"
  python experiments/calibrate_thresholds.py --channel diffpure_gl --t-diffuse $T \
    --purifier-ckpt $CK --sampler ddim --subset $SUBSET --cross-session \
    --n-spk $NSPK --n-utt 4 --tag v7_xsess_tc \
    > $R/calib_diffpure_gl_t${T}_v7_xsess_tc.log 2>&1
  grep "EER" $R/calib_diffpure_gl_t${T}_v7_xsess_tc.log
done

# ---------------------------------------------------------------- 2. channel gate
# The OTHER gate: a purifier that destroys UNPROTECTED identity makes any
# protection measured against it look good for the wrong reason. An earlier
# purifier of ours hit 14-23 % EER on unprotected speech and had to be thrown out.
say "channel gate: EER on unprotected audio must stay near the clean figure"
python - <<'EOF'
import sys, os; sys.path.insert(0, '.')
from utils.csvio import read_csv
clean = read_csv('experiments/results/thresholds_clean_xsess_tc.csv').set_index('encoder')
bad = []
for t in (50, 100, 200):
    p = f'experiments/results/thresholds_diffpure_gl_t{t}_ddim_v7_xsess_tc.csv'
    if not os.path.isfile(p):
        print(f'  t={t}: MISSING'); continue
    d = read_csv(p).set_index('encoder')
    for e in d.index:
        gap = d.eer[e] - clean.eer[e]
        flag = '  <-- CHANNEL DEGRADES IDENTITY' if gap > 0.10 else ''
        print(f'  t={t:<4} {e:<22} EER {100*clean.eer[e]:.2f}% -> {100*d.eer[e]:.2f}%{flag}')
        if gap > 0.10:
            bad.append((t, e))
if bad:
    print('\nFAIL: this purifier damages unprotected identity. Protection measured')
    print('against it would look good for the wrong reason. Do not proceed.')
    sys.exit(1)
print('\nchannel gate PASSED')
EOF
if [ $? -ne 0 ]; then say "FATAL: channel gate failed"; exit 1; fi

say "BASE ESTABLISHED. test-clean + v7 + cross-session thresholds are ready."
say "Next: roadmap P1.3 (loud budget, encoder held out) -- run_p13_loud.sh"
