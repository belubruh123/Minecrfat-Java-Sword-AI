#!/bin/bash
# Boots the headless training server and waits until the bridge is listening.
# Usage: scripts/start_server.sh [logfile]
set -u
LOG="${1:-/tmp/drlagent-server.log}"
cd "$(dirname "$0")/../mod" || exit 1

pkill -f "fabric.dli.main" 2>/dev/null && sleep 2
: > "$LOG"
DRLAGENT_TRAIN=1 ./gradlew runServer > "$LOG" 2>&1 &

for i in $(seq 1 120); do
  if grep -q 'bridge listening' "$LOG" 2>/dev/null; then
    echo "server up (${i}s)"
    exit 0
  fi
  if grep -qE 'BUILD FAILED|FAILURE|Exception in thread "main"' "$LOG" 2>/dev/null; then
    echo "server FAILED to boot:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 1
done
echo "server boot TIMEOUT" >&2
tail -20 "$LOG" >&2
exit 1
