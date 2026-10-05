#!/usr/bin/env bash
# Paper 3 experiment queue (paper/PAPER3_PLAN.md). ONE GPU job at a time.
# Status 2026-09-16: E6 done (clean). E2 v3->v4 done. This script runs the rest.
cd "$(dirname "$0")/.."
say() { echo "[paper3] $(date +%H:%M:%S) $*"; }
wait_gpu() { while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done; }
R=experiments/results
clean() { grep -v -i "warn\|tensorflow\|oneDNN"; }

# ---- E3: a SECOND independently trained purifier ---------------------------
# The 1-D UNet (v5) was abandoned: denoising gain sat at -0.5 for 7k steps, the
# same failure mode as v2. v6 differs from v3/v4 by capacity (base 64 vs 32),
# initialisation (seed 7), and training crops -- an independent training run,
# which is what the transfer claim needs. The beta schedule is shared with
# inference, so it is deliberately NOT varied here.
wait_gpu
if [ ! -f checkpoints/purifier_v6.pt ]; then
  say "E3 training purifier v6 (unet2d base 64, seed 7)"
  python training/train_purifier_v2.py --arch unet2d --base 64 --residual \
    --t-max 200 --lr 5e-4 --steps 30000 --seed 7 --crop 160 \
    --out checkpoints/purifier_v6.pt > checkpoints/train_v6.log 2>&1
fi
GAIN=$(python -c "import torch;print(torch.load('checkpoints/purifier_v6.pt',map_location='cpu').get('val_denoise_gain',-9))" 2>/dev/null || echo -9)
say "v6 denoising gain = $GAIN  (gate: >= 0.15, the v2 lesson)"

# ---- E2: second transfer cell, only if v6 actually purifies ------------------
if python -c "import sys;sys.exit(0 if float('$GAIN')>=0.15 else 1)"; then
  wait_gpu; say "E2 transfer v3 -> v6"
  python experiments/exp_nullspace_survival.py --n-files 20 --spk-offset 20 --utt-index 3 \
    --t 50,100,200 --null-t 200 --seeds 2026,7 --sampler ddim \
    --purifier-ckpt checkpoints/purifier_v3.pt --eval-ckpt checkpoints/purifier_v6.pt \
    --out $R/nullspace_xfer_v3_to_v6.csv > $R/xfer_v6.log 2>&1
  clean < $R/xfer_v6.log | sed -n '/TRANSFER/,$p'
else
  say "v6 FAILED the gate -- not used. Transfer rests on v3->v4 alone."
fi

# ---- E4: survival at n=20, two utterances (replaces the n=5 result) ----------
for U in 3 6; do
  wait_gpu; say "E4 survival, 20 speakers, utterance $U"
  python experiments/exp_nullspace_survival.py --n-files 20 --spk-offset 20 --utt-index $U \
    --t 50,100,200,300 --null-t 100,200 --seeds 2026,7 --sampler ddim \
    --purifier-ckpt checkpoints/purifier_v3.pt \
    --out $R/nullspace_survival_20spk_u${U}.csv > $R/surv20_u${U}.log 2>&1
  clean < $R/surv20_u${U}.log | sed -n '/MEAN over/,$p'
done

# ---- E4b: control -- is the subspace just memorised training speakers? ------
wait_gpu; say "E4b survival on purifier-TRAINING speakers"
python experiments/exp_nullspace_survival.py --n-files 16 --spk-offset 0 --utt-index 3 \
  --t 50,200 --null-t 200 --seeds 2026 --sampler ddim \
  --purifier-ckpt checkpoints/purifier_v3.pt \
  --out $R/nullspace_survival_trainspk.csv > $R/surv_trainspk.log 2>&1
clean < $R/surv_trainspk.log | sed -n '/MEAN over/,$p'

# ---- E5: how big is the subspace? -------------------------------------------
wait_gpu; say "E5 subspace dimension"
python experiments/exp_nullspace_dimension.py --n-files 5 --spk-offset 20 --n-dirs 48 \
  --null-t 200 --t 200 --psr-every 6 --sampler ddim \
  --purifier-ckpt checkpoints/purifier_v3.pt > $R/dimension.log 2>&1
clean < $R/dimension.log | sed -n '/PASS-THROUGH ERROR/,$p'

say "all done"
