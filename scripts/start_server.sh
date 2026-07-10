#!/bin/bash
# Boots the headless training server and waits until the bridge is listening.
# Usage: scripts/start_server.sh [logfile]
set -u
# One log per invocation: a previous backgrounded gradlew writes BUILD FAILED
# into its log when its server JVM is killed, so sharing a path makes the new
# invocation's failure-grep trip over the old epitaph.
LOG="${1:-/tmp/drlagent-server-$(date +%s).log}"
ln -sf "$LOG" /tmp/drlagent-server.log
cd "$(dirname "$0")/../mod" || exit 1

pkill -f "fabric.dli.main" 2>/dev/null
# let the old wrapper exit so two runServer builds don't queue on the daemon
for i in $(seq 1 30); do
  pgrep -f "gradlew runServer|GradleWrapperMain.*runServer" >/dev/null || break
  sleep 1
done
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
