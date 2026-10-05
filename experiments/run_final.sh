#!/usr/bin/env bash
# Re-run the on-manifold probe with the signal-gain metric, so the
# double-dissociation table in the paper uses ONE survival metric for both arms
# (see RESULTS section 11a: a norm ratio cannot separate "survived" from
# "destroyed and replaced by equal-sized noise").
cd "$(dirname "$0")/.."
R=experiments/results
until grep -q "^\[e7\] done" $R/e7_queue.log 2>/dev/null; do sleep 20; done
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 15; done
echo "[final] $(date +%H:%M:%S) parametric probe with signal gain"
python experiments/exp_parametric_probe.py --n-files 10 --spk-offset 20 \
  --t 50,200 --transforms tilt,eq,warp,reverb,mcadams \
  --purifier-ckpt checkpoints/purifier_v3.pt \
  --out $R/parametric_probe_alpha.csv > $R/probe_alpha.log 2>&1
grep -v -i "warn\|tensorflow\|oneDNN" $R/probe_alpha.log | tail -40
echo "[final] done"
