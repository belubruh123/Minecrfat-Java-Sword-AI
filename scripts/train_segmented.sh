#!/bin/bash
# Segmented training: reboot the server every SEG steps and resume the run.
# Long single-boot sessions accumulate server-side state that eventually
# corrupts the environment: armor durability (fixed by fresh kits) was one
# instance; a second corruption breaks HIT REGISTRATION after ~80k steps on a
# single boot (swings stop landing -> value function sees only timeouts ->
# entropy drifts up -> the policy dissolves). A continuous 80k-step duel2 run
# hit exactly this. Fresh boots are cheap (~40 s); training resumes from
# runs/<run>/latest.pt with optimizer state intact.
#
# Hardened: force-kills the old JVM, WAITS for the bridge port (36565) to free
# before rebooting (a lingering bind was the old exit-1), and retries a failed
# segment once on a fresh boot instead of aborting the whole ladder.
#
# Usage: scripts/train_segmented.sh <config.yaml> <run-name> <total-steps> [seg-steps]
# Optional env vars (used only on the FIRST segment, to warm-start a NEW run
# from a previous stage — later segments resume this run's own latest.pt):
#   INIT_CKPT=../runs/<prev>/best.pt   warm-start weights (fresh optimizer)
#   INIT_CKPT=none                     train from scratch (ignore config init)
#   AIM_CKPT=../models/aim.pt          override the frozen aim net
set -u
CFG="$1"; RUN="$2"; TOTAL="$3"; SEG="${4:-40000}"
cd "$(dirname "$0")/.." || exit 1
LOG="/tmp/drlagent-server-$RUN.log"
PORT=36565
INIT_CKPT="${INIT_CKPT:-}"
AIM_CKPT="${AIM_CKPT:-}"

wait_port_free() {
  for _ in $(seq 1 30); do
    ss -tlnp 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 1
  done
  return 0  # proceed anyway; start_server does its own bind
}

boot() {
  pkill -9 -f "fabric.dli.main" 2>/dev/null
  wait_port_free
  scripts/start_server.sh "$LOG"
}

# Resuming a run already past the early segment targets? Start the segment
# counter at the checkpoint's step (rounded down to a SEG multiple). Otherwise
# we reboot the server dozens of times just to have train.py exit immediately
# because global_step already exceeds a tiny 40k segment target — minutes of
# churn with zero training. Only affects runs with an existing latest.pt.
CUR=0
if [ -f "runs/$RUN/latest.pt" ]; then
  STEP=$(.venv/bin/python -c "import torch;print(torch.load('runs/$RUN/latest.pt',map_location='cpu',weights_only=False).get('step',0))" 2>/dev/null || echo 0)
  case "$STEP" in ''|*[!0-9]*) STEP=0 ;; esac
  CUR=$(( (STEP / SEG) * SEG ))
  [ "$CUR" -gt 0 ] && echo "resuming $RUN at step $STEP -> segments start at $CUR (skipping no-op reboots)"
fi
while [ "$CUR" -lt "$TOTAL" ]; do
  CUR=$((CUR + SEG)); [ "$CUR" -gt "$TOTAL" ] && CUR="$TOTAL"
  RES=""
  if [ -f "runs/$RUN/latest.pt" ]; then
    RES="--resume ../runs/$RUN/latest.pt"          # continue this run
  elif [ -n "$INIT_CKPT" ]; then
    RES="--init-checkpoint $INIT_CKPT"             # first segment: warm-start
  fi
  [ -n "$AIM_CKPT" ] && RES="$RES --aim-checkpoint $AIM_CKPT"
  echo "=== segment -> step $CUR / $TOTAL ==="
  ok=0
  for attempt in 1 2; do
    if ! boot; then
      echo "server boot failed (attempt $attempt), retrying" >&2
      continue
    fi
    if (cd trainer && ../.venv/bin/python train.py "$CFG" --run "$RUN" \
        --steps "$CUR" $RES); then
      ok=1; break
    fi
    echo "segment failed (attempt $attempt), fresh boot + retry" >&2
  done
  [ "$ok" = 1 ] || { echo "segment aborted after retries at step $CUR" >&2; exit 1; }
done
echo "segmented training complete: $RUN @ $TOTAL steps"
