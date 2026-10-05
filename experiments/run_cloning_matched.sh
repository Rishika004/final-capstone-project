#!/usr/bin/env bash
# The MATCHED-purifier cloning condition, to complete the mismatch comparison.
# The first run covered `none` (no purification) and `diffwave` (foreign purifier).
# Section 5 of the paper is about MATCHED vs MISMATCHED, so we also need `ours`
# -- the purifier the perturbations were actually crafted against. Without it the
# cloning arm cannot test whether the paper's central finding replicates.
cd "$(dirname "$0")/.."
while tasklist 2>/dev/null | grep -qi python.exe; do sleep 20; done
echo "[clone-matched] $(date +%H:%M:%S) ours (v3, t=200) as the attacker's purifier"
python experiments/exp_cloning_protection.py --n-files 20 --purifiers ours \
  --out experiments/results/cloning_protection_ours.csv \
  > experiments/results/cloning_protection_ours.log 2>&1
echo "[clone-matched] done"
