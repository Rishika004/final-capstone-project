#!/usr/bin/env bash
# ============================================================================
# The W7 GPU queue: everything still owed, in the order its value demands.
#
# ONE GPU JOB AT A TIME is a project rule (RESULTS 31: a probe run beside a
# crafting arm OOMed it). This script waits for the 47e arm to finish, then
# runs the remaining pre-registered work back to back so the GPU is never
# idle between jobs.
#
# ORDER, and why:
#   1. 47e held-out seg  -- already running; mandatory before C2 can be
#                           written either way.
#   2. CONTROL B         -- the only open item that can FALSIFY A HEADLINE
#                           (it can take "speaker" out of the title), so it
#                           goes before anything that merely strengthens.
#   3. #13c              -- C1's last structural hole.
#
# Each stage runs only if its predecessor produced what it promised, and each
# analyser runs immediately after its stage so a failure is visible in the log
# rather than three hours later.
#
# Run:  nohup bash experiments/run_queue_w7.sh > experiments/results/queue_w7.log 2>&1 &
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1

R=experiments/results
say() { echo "[w7] $(date +%H:%M:%S) $*"; }

# ------------------------------------------------- 1. wait for the 47e arm
SEG_DONE=1
for f in heldout_seg_c2_libri_t1 heldout_seg_c2_libri_t3 \
         heldout_seg_c2_sc09_t1 heldout_seg_c2_sc09_t3; do
  [ -f "$R/$f.csv" ] || SEG_DONE=0
done

# Liveness is judged by LOG STALENESS, not by pgrep -- this Git Bash has no
# pgrep, and an earlier version of this script silently treated "pgrep: command
# not found" as "the job is dead".
STALE_LIMIT=1200          # 20 min with no log write = dead (a speaker takes ~6)

newest_mtime() {           # most recent mtime across the runner's logs
  local newest=0 m
  for f in "$R"/heldout_seg_craft.log "$R"/heldout_seg_runner.log \
           "$R"/heldout_seg_libri_t1.log "$R"/heldout_seg_libri_t3.log \
           "$R"/heldout_seg_sc09_t1.log "$R"/heldout_seg_sc09_t3.log; do
    [ -f "$f" ] || continue
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$m" -gt "$newest" ] && newest=$m
  done
  echo "$newest"
}

if [ "$SEG_DONE" = "0" ]; then
  say "waiting for the 47e held-out seg arm (run_heldout_seg.sh)"
  WAITED=0
  while [ "$WAITED" -lt 36000 ]; do          # 10 h ceiling, then give up
    SEG_DONE=1
    for f in heldout_seg_c2_libri_t1 heldout_seg_c2_libri_t3 \
             heldout_seg_c2_sc09_t1 heldout_seg_c2_sc09_t3; do
      [ -f "$R/$f.csv" ] || SEG_DONE=0
    done
    [ "$SEG_DONE" = "1" ] && break

    NOW=$(date +%s); M=$(newest_mtime)
    if [ "$M" -gt 0 ] && [ $((NOW - M)) -gt "$STALE_LIMIT" ]; then
      say "no log write in $((NOW - M))s (limit ${STALE_LIMIT}s) and outputs"
      say "incomplete -- the 47e arm appears to have stalled or died."
      say "Continuing to Control B, which is independent of that arm."
      break
    fi
    sleep 120
    WAITED=$((WAITED + 120))
  done
  [ "$WAITED" -ge 36000 ] && say "10 h ceiling reached; moving on"
fi

if [ "$SEG_DONE" = "1" ]; then
  say "47e arm complete -- reading it"
  python experiments/analyse_heldout_seg.py > $R/heldout_seg_verdict.log 2>&1
  say "analyse_heldout_seg exit $?; verdict in heldout_seg_verdict.log"
  tail -25 $R/heldout_seg_verdict.log
else
  say "47e arm NOT complete; its verdict is not available"
fi

# ------------------------------------------------------------ 2. CONTROL B
if [ -f "$R/encoder_subspace_controlb.csv" ]; then
  say "Control B output already exists -- skipping (B.4 forbids re-running)"
else
  say "CONTROL B -- can remove 'speaker' from the title"
  bash experiments/run_control_b.sh
  say "run_control_b exit $?"
  if [ -f "$R/encoder_subspace_controlb.csv" ]; then
    python experiments/analyse_control_b.py > $R/control_b_verdict.log 2>&1
    say "analyse_control_b exit $?"
    tail -30 $R/control_b_verdict.log
  else
    say "Control B produced no output -- see $R/control_b.log"
    tail -20 $R/control_b.log 2>/dev/null
  fi
fi

# ---------------------------------------------------------------- 3. #13c
if [ -f "$R/encoder_subspace_wav.csv" ]; then
  say "#13c output already exists -- skipping"
else
  say "#13c -- alpha through DiffWave, C1's last structural hole"
  bash experiments/run_13c.sh
  say "run_13c exit $?"
  if [ -f "$R/encoder_subspace_wav.csv" ]; then
    python experiments/analyse_13c.py > $R/13c_verdict.log 2>&1
    say "analyse_13c exit $?"
    tail -30 $R/13c_verdict.log
  else
    say "#13c produced no output -- see $R/13c_libri.log"
    tail -20 $R/13c_libri.log 2>/dev/null
  fi
fi

say "QUEUE COMPLETE."
say "verdicts: heldout_seg_verdict.log, control_b_verdict.log, 13c_verdict.log"
