#!/usr/bin/env bash
# experiments/skipguard.sh -- "rows exist" is not "this arm is done".  (PLAN_V3 W0)
#
# Source this file, then replace
#     if grep -q "^$tag," $OUT 2>/dev/null; then say "skip $tag"; return; fi
# with
#     if skip_guard "$tag" "$OUT"; then say "skip $tag"; return; fi
#
# skip_guard returns 0 (skip) only when rows tagged TAG already exist in OUT
# AND every one of them carries the code_sha of the code that is about to run.
# It returns 1 (run) when no such rows exist.  When rows exist but were written
# by a different code version -- or by code that predates the code_sha column
# (everything up to tag recipe-v1) -- it REFUSES and exits the whole script:
# the old guard would have silently merged arms crafted by two code versions
# into one CSV, which is the RESULTS §25 failure mode.
#
# Escape hatch, deliberate and loud:  SKIPGUARD_FORCE_SKIP=1  accepts the stored
# rows as-is and prints what it accepted.  Use it for re-running the eval-only
# stages of a script whose crafting rows are the frozen v1 base.

skip_guard() {
  local tag="$1" out="$2" here cur stored
  [ -n "$tag" ] && [ -n "$out" ] || { echo "[skipguard] usage: skip_guard TAG OUT" >&2; exit 2; }
  [ -f "$out" ] || return 1
  grep -q "^$tag," "$out" 2>/dev/null || return 1
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cur="$(python "$here/utils/provenance.py")" || { echo "[skipguard] cannot read code_sha" >&2; exit 2; }
  stored="$(python "$here/utils/provenance.py" --stored "$out" "$tag")"
  if [ "$stored" = "$cur" ]; then
    return 0
  fi
  case "$cur" in *-dirty*)
    echo "[skipguard] NOTE: working tree has uncommitted code changes (code_sha=$cur)" >&2 ;;
  esac
  if [ "${SKIPGUARD_FORCE_SKIP:-0}" = "1" ]; then
    echo "[skipguard] FORCED skip of '$tag' in $out: stored code_sha='$stored', current='$cur'" >&2
    return 0
  fi
  cat >&2 <<MSG
[skipguard] FATAL: refusing to skip '$tag'.
  rows tagged '$tag' already exist in $out
  stored code_sha : $stored   (MISSING = written before the column existed, i.e. <= recipe-v1)
  current code_sha: $cur
  Skipping would merge arms crafted by two code versions into one file.
  Either: (a) write to a new --out, (b) delete those rows, or
          (c) SKIPGUARD_FORCE_SKIP=1 to accept the stored rows as the frozen base.
MSG
  exit 1
}
