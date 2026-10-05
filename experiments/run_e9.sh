#!/usr/bin/env bash
# E9: does the pass-through subspace transfer to a purifier we did NOT train?
# Two public unconditional DiffWave checkpoints, both foreign in architecture
# (WaveNet on raw waveform), weights, data and domain:
#   libri = De-AntiFake (Fan et al. ICML 2025), trained on LibriSpeech -- the
#           purifier from the paper that motivates ours. The one that matters.
#   sc09  = philsyn, trained on Speech Commands digits. Narrow prior, so a null
#           against this one alone could be a domain artefact.
# t is calibrated by DAMAGE, not by integer: DiffWave t=1 gives mel damage
# 0.29-0.31 against our v3 t=200 at 0.2933; t=3 (AudioPure's default) is already
# 0.44-0.48, i.e. harsher than our strongest setting. Both are run.
cd "$(dirname "$0")/.."
R=experiments/results
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done
for CK in libri_deantifake sc09_philsyn; do
  for T in 1 3; do
    echo "[e9] $(date +%H:%M:%S) foreign=$CK t=$T"
    python experiments/exp_foreign_purifier.py --foreign diffwave \
      --foreign-ckpt checkpoints/foreign/diffwave_${CK}.pkl --foreign-t $T \
      --n-files 10 --spk-offset 20 --utt-index 3 --t 200 --null-t 200 \
      --sampler ddim --vocoder gl \
      --out $R/foreign_${CK}_t${T}.csv > $R/e9_${CK}_t${T}.log 2>&1
    grep -viE "warn|weight_norm|RealDiffPure|detach" $R/e9_${CK}_t${T}.log | sed -n '/FOREIGN-PURIFIER/,$p'
  done
done
echo "[e9] done"
