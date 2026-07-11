# Setup Guide

Two ways to use this project:

- **[A. Pilot mode](#a-pilot-mode--use-the-pretrained-agent)** — let the
  pretrained agent fight for you. No training, no server, ~5 minutes.
- **[B. Training stack](#b-training-stack--optional)** — retrain or extend
  the models yourself. Only needed if you want to change how the agent plays.

Both need the same one-time backend setup.

## Prerequisites

| What | Version | Needed for |
|---|---|---|
| Minecraft + Fabric Loader ≥ 0.19.3 + Fabric API | 26.1.2 | playing with the mod |
| Python | 3.12+ | the backend (inference + training) |
| Java (JDK) | 25 | only for building the mod / running the training server |

If you use the prebuilt jar and pretrained models, you do **not** need Java —
only Python.

## One-time backend setup

From the repo root (works the same on Linux/macOS; Windows notes below):

```bash
python3 -m venv .venv
.venv/bin/pip install -r trainer/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index gets you the CPU-only PyTorch wheel (~200 MB instead of
several GB of CUDA — inference is tiny, no GPU needed).

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r trainer\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Wherever this guide says `../.venv/bin/python`, use `..\.venv\Scripts\python`
on Windows.

## A. Pilot mode — use the pretrained agent

What you need on the machine you play Minecraft on:

1. **The mod jar** — `mod/build/libs/drlagent-0.1.0.jar` (or build it:
   `cd mod && ./gradlew build`). Drop it in your Minecraft `mods/` folder next
   to Fabric API, like any other mod.
2. **This repo's `trainer/` and `models/` folders** — anywhere on disk, plus
   the backend setup above. `models/` ships the trained weights
   (`aim.pt`, `swing.pt`, `move.pt`, and `combo.pt` once stage 5 finishes) —
   you never need to train.

Then, each time you play:

```bash
cd trainer
../.venv/bin/python pilot.py        # "pilot server listening on 127.0.0.1:36566"
```

Leave that running, start Minecraft, join a world/server, and:

| In game | What it does |
|---|---|
| **G** key (rebindable: "Toggle DRL Pilot") | engage / release the pilot |
| `/pilot on` / `/pilot off` | same as the key |
| `/pilot status` | engaged?, current target, server port |
| `/pilot port 12345` | use a different pilot.py port (`pilot.py --port 12345`) |

While engaged the pilot locks onto the nearest player (within 48 blocks),
aims, times swings, and fights. A stats panel in the top-left shows the
fight live: hits landed (split into sprint hits and crits), the current
combo chain with best/total combos, and hits taken — counters reset each
time you engage. The pilot releases instantly when you toggle it off, and
auto-releases if it leaves the world or `pilot.py` stops answering — you
are never locked out of your own character.

`pilot.py` options: `--aim/--swing/--move` point at other checkpoints
(omitting one disables that head — useful for testing aim only), `--port`
changes the listen port. It prefers `models/combo.pt` over `models/move.pt`
automatically when both exist.

**Fair-play warning**: the pilot only sees what a player could see and only
acts through vanilla input paths, but on public servers it is still botting
and will get the account banned. Use it on your own server or with consenting
opponents.

## B. Training stack — optional

Everything runs on one machine, CPU-only (tested on 4 cores / 5.8 GB RAM).
Java 25 required.

```bash
scripts/start_server.sh     # builds + boots the headless training server,
                            # returns when the bridge (:36565) is up
cd trainer
../.venv/bin/python train.py configs/stage5_combo.yaml --run stage5_combo
```

- Stages build on each other: `stage1_aim` → `stage2_vertical` →
  `stage3_swing` → `stage4_move` → `stage5_combo`. Each config names the
  frozen earlier checkpoints (`aim_checkpoint`, `swing_checkpoint`), so
  train them in order — or just reuse the shipped `models/*.pt` and only
  train the stage you care about.
- Interrupt any time; resume with
  `--resume ../runs/<run>/latest.pt`.
- Watch training: replay dashboard
  `../.venv/bin/python dashboard/app.py` → http://localhost:8080 (replays
  arena 0 at real speed, exactly what the agent sees), and TensorBoard
  `../.venv/bin/tensorboard --logdir ../runs`.

Evaluate a checkpoint (stop the trainer first — the bridge is single-client):

```bash
../.venv/bin/python eval.py ../runs/stage5_combo/final.pt --episodes 100 \
    --curriculum configs/stage5_combo.yaml \
    --aim ../models/aim.pt --swing ../models/swing.pt --stochastic
```

To make the pilot use a model you trained, copy it into `models/`
(e.g. `cp runs/stage5_combo/final.pt models/combo.pt`) and restart `pilot.py`.

## Ports (all localhost-only)

| Port | Who listens | Used for |
|---|---|---|
| 36565 | mod (training server) | lock-step training bridge |
| 36566 | `pilot.py` | pilot inference |
| 8080 | dashboard | episode replays |
| 25565 | training server | normal Minecraft joins (e.g. to spectate) |
