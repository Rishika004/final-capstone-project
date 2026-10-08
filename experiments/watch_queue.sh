#!/usr/bin/env bash
# ============================================================================
# A TRACKED watcher over the untracked GPU chain.
#
# The chain itself (run_queue_w7.sh, run_queue_w7b.sh) was launched with
# `nohup ... &`, which detaches it from the Claude Code harness: no sidebar
# entry, no progress, no completion notification. The jobs run fine, but they
# are invisible.
#
# Killing and relaunching them under tracking would throw away whatever is
# mid-flight, so instead this script WATCHES them and is itself tracked. It
# touches nothing, holds no GPU, and exits when the chain is done -- which is
# what produces the notification.
#
# Run (from the harness, tracked):  bash experiments/watch_queue.sh
# ============================================================================
cd "/c/Users/Rishika/OneDrive/Desktop/Phantom/nullspace_voice_protection" || exit 1
R=experiments/results
say() { echo "[watch] $(date +%H:%M:%S) $*"; }

STALE=2400          # 40 min with no log write anywhere = the chain is stuck
LAST_REPORT=""

newest() {          # seconds since the most recent write by any chain log
  local newest=0 m now
  for f in "$R"/control_b.log "$R"/13c_libri.log "$R"/13c_sc09.log \
           "$R"/b7_protected_asv.log "$R"/queue_w7.log "$R"/queue_w7b.log; do
    [ -f "$f" ] || continue
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$m" -gt "$newest" ] && newest=$m
  done
  now=$(date +%s)
  echo $((now - newest))
}

progress() {
  local p=""
  if [ -f "$R/control_b_verdict.log" ]; then
    p="ControlB done"
  elif [ -f "$R/control_b.log" ]; then
    p="ControlB $(grep -o '\[[0-9]*/29\]' "$R/control_b.log" | tail -1)"
  fi
  if [ -f "$R/13c_verdict.log" ]; then
    p="$p | 13c done"
  elif [ -f "$R/13c_libri.log" ] || [ -f "$R/13c_sc09.log" ]; then
    local n
    n=$(cat "$R"/13c_*.log 2>/dev/null | grep -c '^\[13c\] [0-9]*/29')
    p="$p | 13c ${n} measurements"
  fi
  [ -f "$R/asv_protected_arm.csv" ] && p="$p | B7 done"
  echo "$p"
}

say "watching the detached GPU chain: Control B -> #13c -> B7"
say "this watcher holds no GPU and changes nothing"

WAITED=0
while [ "$WAITED" -lt 72000 ]; do          # 20 h ceiling
  if [ -f "$R/asv_protected_arm.csv" ] \
     && grep -q "W7b COMPLETE" "$R/queue_w7b.log" 2>/dev/null; then
    say "CHAIN COMPLETE"
    break
  fi
  S=$(newest)
  if [ "$S" -gt "$STALE" ]; then
    say "STALLED: no chain log written for ${S}s (limit ${STALE}s)"
    say "last known progress: $(progress)"
    exit 2
  fi
  CUR="$(progress)"
  if [ "$CUR" != "$LAST_REPORT" ]; then
    say "$CUR"
    LAST_REPORT="$CUR"
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

echo
say "================= VERDICTS ================="
for v in control_b_verdict 13c_verdict; do
  if [ -f "$R/$v.log" ]; then
    echo "--- $v"
    sed -n '/VERDICT/,$p' "$R/$v.log" | head -24
  else
    say "$v.log absent"
  fi
done
if [ -f "$R/b7_protected_asv.log" ]; then
  echo "--- B7"
  tail -14 "$R/b7_protected_asv.log"
fi
say "done"
