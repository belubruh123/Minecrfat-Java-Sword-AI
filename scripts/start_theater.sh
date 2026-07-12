#!/bin/bash
# Replay theater: a real headless Minecraft client that re-enacts the newest
# recorded episodes on the training server's arena and streams its screen
# into the dashboard ("Live Minecraft render" panel).
#
# Usage: scripts/start_theater.sh [run-name]     (default stage6_fighter)
#        scripts/start_theater.sh stop
#
# Costs roughly one CPU core (software GL) — stop it when not watching to
# give training its full speed back.
set -u
cd "$(dirname "$0")/.." || exit 1
RUN="${1:-stage6_fighter}"

if [ "$RUN" = "stop" ]; then
  # bracket patterns so pkill does not match this script's own command line
  pkill -f "PilotTest[e]r"
  pkill -f "x11grab.*liv[e].jpg"
  rm -f runs/live.jpg
  echo "theater stopped"
  exit 0
fi

pgrep -f "Xvf[b] :99" >/dev/null || { Xvfb :99 -screen 0 1280x720x24 & sleep 2; }
export DISPLAY=:99

if ! pgrep -f "PilotTest[e]r" >/dev/null; then
  rm -f mod/run/pilot-client/logs/latest.log
  (cd mod && ./gradlew runPilotClient > /tmp/drlagent-theater.log 2>&1 &)
  echo "booting theater client…"
  for i in $(seq 1 120); do
    grep -q "Connecting to localhost" mod/run/pilot-client/logs/latest.log 2>/dev/null && break
    sleep 3
  done
  sleep 12   # let the world finish loading
fi

xdotool key slash; sleep 1
xdotool type --delay 40 "replay follow $RUN"; sleep 0.5
xdotool key Return; sleep 2

# grab exactly the game window (it does not fill the virtual display)
GEO=$(xdotool search --name "Minecraft" getwindowgeometry --shell 2>/dev/null | head -5)
X=$(echo "$GEO" | sed -n 's/^X=//p'); Y=$(echo "$GEO" | sed -n 's/^Y=//p')
W=$(echo "$GEO" | sed -n 's/^WIDTH=//p'); H=$(echo "$GEO" | sed -n 's/^HEIGHT=//p')
pgrep -f "x11grab" >/dev/null || \
  ffmpeg -loglevel error -f x11grab -video_size "${W:-640}x${H:-360}" -framerate 10 \
         -i ":99.0+${X:-0},${Y:-0}" -q:v 6 -update 1 -y runs/live.jpg &
sleep 3
grep "\[replay\]" mod/run/pilot-client/logs/latest.log | tail -2
echo "theater up — watch the dashboard's 'Live Minecraft render' panel"
