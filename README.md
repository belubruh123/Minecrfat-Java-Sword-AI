# DRL Sword-PvP Agent (Minecraft 26.1.2, Fabric)

Trains a deep-RL agent to play diamond-kit sword PvP from human-fair
observations: a 426×240 binary silhouette of the opponent (what a player
sees, computed geometrically — no rendering) plus the scalars a human also
knows (own speed, yaw/pitch, height, cooldown, health, last-hit reach).
Policies are trained cumulatively, earlier ones frozen: **aim**
(Δyaw/Δpitch) → **swing** (attack timing + reach) → **move** (W key /
sprint spacing) → **combo** (sprint-hit knockback chains + jump crits).
See `PROTOCOL.md` for the wire format and **`SETUP.md` for step-by-step
install/usage instructions** (backend, pilot, training).

## Layout

- `mod/` — Fabric mod. Headless training server: 8 parallel arenas of
  FakePlayers, lock-step tick bridge on :36565, arena-0 episode recorder.
  Client side: pilot mode (below).
- `trainer/` — Python PPO trainer (`train.py`), eval suites (`eval.py`),
  replay dashboard (`dashboard/`, :8080), pilot inference server (`pilot.py`).
- `runs/` — checkpoints, metrics, recorded episodes.

## Training

```bash
scripts/start_server.sh                 # boots the dev server, waits for the bridge
cd trainer
../.venv/bin/python train.py configs/stage1_aim.yaml --run stage1_aim
../.venv/bin/python dashboard/app.py    # watch replays at http://localhost:8080
```

Stages 2–4 chain checkpoints via their configs (`aim_checkpoint`,
`swing_checkpoint`). Evaluate with the bridge free (trainer stopped):

```bash
../.venv/bin/python eval.py ../runs/stage3_swing/latest.pt --episodes 100 \
    --curriculum configs/stage3_swing.yaml --aim ../runs/stage2_vertical_v2/best.pt --stochastic
# move checkpoints additionally take --swing ../runs/stage3_swing/latest.pt
```

## Pilot mode — let the trained agent fight for you

The client half of the mod can hand your own character to the trained
policies. It observes through the same code the agent trained on and acts
only through vanilla input paths (key mappings, the normal attack packet) —
it can do nothing a human at the keyboard could not.

The trained weights ship in `models/` (aim.pt, swing.pt, move.pt) — no
training needed to use the pilot.

1. Build the mod: `cd mod && ./gradlew build` → `mod/build/libs/drlagent-*.jar`
   (the training server must be stopped first; it holds the Gradle lock).
2. Install the jar in your client's `mods/` folder (Fabric Loader ≥ 0.19.3
   + Fabric API for MC 26.1.2), together with a copy of `trainer/` and
   `models/` on the same machine.
3. Start the inference server: `cd trainer && python pilot.py`
   (defaults to the `models/` checkpoints; point `--aim`/`--swing`/`--move`
   at other files for staged testing — a missing swing/move file just
   disables that head).
4. In game, press **G** (rebindable: "Toggle DRL Pilot") or use the commands:
   `/pilot on`, `/pilot off`, `/pilot status`, `/pilot port <n>`. The pilot
   locks on to the nearest player and fights; toggle again to take back
   control. It auto-disengages if the inference server stops answering.

Use it only where it's allowed — on your own server or with consenting
opponents. On public servers this is indistinguishable from botting and
will get the account banned.
