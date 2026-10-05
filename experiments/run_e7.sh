#!/usr/bin/env bash
# E7 cross-example transfer + the second purifier, after the main queue is done.
cd "$(dirname "$0")/.."
R=experiments/results
until grep -q "all done" $R/paper3_queue.log 2>/dev/null; do sleep 20; done
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done
echo "[e7] $(date +%H:%M:%S) cross-example transfer"
python experiments/exp_crossexample_transfer.py --n-files 6 --spk-offset 20 \
  --null-t 200 --t 200 --sampler ddim --purifier-ckpt checkpoints/purifier_v3.pt \
  > $R/crossexample.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" $R/crossexample.log | sed -n '/CROSS-EXAMPLE/,$p'

echo "[e7] $(date +%H:%M:%S) training purifier v6 (second transfer cell)"
python training/train_purifier_v2.py --arch unet2d --base 64 --residual \
  --t-max 200 --lr 5e-4 --steps 20000 --seed 7 --crop 160 \
  --out checkpoints/purifier_v6.pt > checkpoints/train_v6.log 2>&1
GAIN=$(python -c "import torch;print(torch.load('checkpoints/purifier_v6.pt',map_location='cpu').get('val_denoise_gain',-9))" 2>/dev/null || echo -9)
echo "[e7] v6 denoising gain = $GAIN (gate >= 0.15)"
if python -c "import sys;sys.exit(0 if float('$GAIN')>=0.15 else 1)"; then
  python experiments/exp_nullspace_survival.py --n-files 20 --spk-offset 20 --utt-index 3 \
    --t 50,100,200 --null-t 200 --seeds 2026,7 --sampler ddim \
    --purifier-ckpt checkpoints/purifier_v3.pt --eval-ckpt checkpoints/purifier_v6.pt \
    --out $R/nullspace_xfer_v3_to_v6.csv > $R/xfer_v6.log 2>&1
  grep -v -i "warn\|tensorflow\|oneDNN" $R/xfer_v6.log | sed -n '/TRANSFER/,$p'
else
  echo "[e7] v6 failed the gate; transfer rests on v3->v4"
fi
echo "[e7] done"
