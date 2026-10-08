#!/usr/bin/env bash
# ============================================================================
# B7 -- the protected-arm operating point. R2's top ask (council v4 B).
#
# Eval only: no crafting, no purifier. It embeds 29 already-crafted wavs and
# their clean originals against the clean cross-session enrolments, so it costs
# seconds of GPU. It still waits for run_queue_w7.sh rather than running
# beside it, because "one GPU job at a time" is a project rule that exists for
# a measured reason (RESULTS 31: a probe beside a crafting arm OOMed it), and
# a rule with an exception for small jobs is not a rule.
#
# Run: nohup bash experiments/run_queue_w7b.sh > experiments/results/queue_w7b.log 2>&1 &
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1
R=experiments/results
say() { echo "[w7b] $(date +%H:%M:%S) $*"; }

say "waiting for run_queue_w7.sh to print QUEUE COMPLETE"
WAITED=0
while [ "$WAITED" -lt 54000 ]; do           # 15 h ceiling
  grep -q "QUEUE COMPLETE" "$R/queue_w7.log" 2>/dev/null && break
  sleep 120; WAITED=$((WAITED + 120))
done
grep -q "QUEUE COMPLETE" "$R/queue_w7.log" 2>/dev/null \
  || { say "W7 queue never completed; running B7 anyway (it is independent)"; }

if [ -f "$R/asv_protected_arm.csv" ]; then
  say "B7 output exists -- skipping"
else
  say "B7: protected arm as an ASV channel, arm fullnsT_tc_m30_seg"
  python experiments/eval_protected_asv.py --tag fullnsT_tc_m30_seg \
    > $R/b7_protected_asv.log 2>&1
  say "exit $?"
  tail -22 $R/b7_protected_asv.log
fi
say "W7b COMPLETE."
