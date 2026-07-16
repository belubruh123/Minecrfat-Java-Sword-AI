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
   the backend setup above. `models/` ships the trained weights — the pilot
   automatically picks the newest generation it finds
   (`fighter2.pt` > `fighter.pt` > `combo.pt` > `move.pt`, plus `aim.pt`
   and `swing.pt`) — you never need to train.

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
changes the listen port. It prefers the newest movement generation automatically
(`fighter2.pt` first). Human-natural output is on by default:
aim smoothing (the raw policy is accurate but visibly shaky) and input
humanizing (keys held at human timescales, clicks jittered and capped at
human rates instead of 20 frame-perfect decisions per second).
`--raw-aim` / `--raw-keys` disable them.

**Fair-play warning**: the pilot only sees what a player could see and only
acts through vanilla input paths, but on public servers it is still botting
and will get the account banned. Use it on your own server or with consenting
opponents.

## B. Training stack — optional

Everything runs on one machine, CPU-only (tested on 4 cores / 5.8 GB RAM).
Java 25 required. **The pretrained models already ship in `models/` — you only
need this if you want to train your own bot.**

### Train your own bot — one command

```bash
scripts/train_all.sh            # full budgets — best model (~12h on CPU)
scripts/train_all.sh quick      # decent model (~4-5h)
scripts/train_all.sh smoke      # tiny — just proves the pipeline runs
```

That is it. It boots and reboots the server for you, trains all five stages
(aim → vertical aim → combo school → mortal fight → **TheoBaldTheBird-style
bot**), and **ships `models/aim.pt` + `models/fighter2.pt`** when done (old ones
backed up to `*_prev.pt`). The final stage hardens the fighter against the
native `theobald` opponent (a faithful 6-difficulty reimplementation of
Theobald's sword bot). The pilot auto-uses the shipped weights — press **G** in
game. Set your own budgets with env vars:

```bash
AIM_STEPS=1000000 VERT_STEPS=1500000 COMBO_STEPS=1500000 \
    FIGHT_STEPS=250000 THEO_STEPS=400000 scripts/train_all.sh
```

Interrupted? Re-run the same command — each stage resumes where it left off.

### Or drive a single stage by hand

```bash
scripts/start_server.sh     # builds + boots the headless training server,
                            # returns when the bridge (:36565) is up
cd trainer
../.venv/bin/python train.py configs/stage7a_comboschool.yaml --run my_combo
```

- **What the one-command script runs**: `stage1_aim` → `stage2_vertical` →
  `stage7a_comboschool` → `stage7b_chain` — see `configs/README.md` for the
  manual order, flags and how to ship. To fine-tune a single stage from the
  shipped `models/*.pt`, run just that stage's config. The `fighter2` stages
  own the attack button themselves and need no `swing_checkpoint`. Superseded
  generations live in `configs/archive/`.
- Interrupt any time; resume with
  `--resume ../runs/<run>/latest.pt`.
- Watch training: replay dashboard
  `../.venv/bin/python dashboard/app.py` → http://localhost:8080 (replays
  arena 0 at real speed: the agent's mask view, a reconstructed 3D view,
  and a top-down tactical map), and TensorBoard
  `../.venv/bin/tensorboard --logdir ../runs`.
- Real Minecraft rendering: `scripts/start_theater.sh [run]` boots a
  headless game client that re-enacts the newest recorded episodes on the
  arena (camera rides the agent, the opponent is a real player model) and
  streams its screen into the dashboard's "Live Minecraft render" panel.
  Costs about a CPU core — `scripts/start_theater.sh stop` when not
  watching. The `/replay follow <run>` command also works on your own
  client (`/replay api <url>` points it at a remote dashboard).

Evaluate a checkpoint (stop the trainer first — the bridge is single-client):

```bash
../.venv/bin/python eval.py ../runs/stage7b_chain/final.pt --episodes 100 \
    --curriculum configs/stage7b_chain.yaml \
    --aim ../models/aim.pt --stochastic
```

To make the pilot use a model you trained, copy it into `models/`
(e.g. `cp runs/stage7b_chain/final.pt models/fighter2.pt`) and restart
`pilot.py`.

### C. Train in-game — "pilot mode, but it learns"

Fine-tune the fighter live, from real play, against whatever opponent is in the
world — including the **real** TheoBaldTheBird HeroBot bot. Same PPO as the
headless trainer, but n=1 and at real time, so it's for **fine-tuning** the
shipped `models/fighter2.pt`, not training from scratch. Only the combat head
learns; the aim net stays frozen.

```bash
cd trainer
../.venv/bin/python pilot_train.py            # listens on :36566, resumes models/fighter2.pt
```

Then in a Minecraft client with the drlagent mod: load a world with an opponent,
target it, and press **G**. The HUD shows the fight stats while the policy
learns from every exchange. Press **G** again (or Ctrl+C in the terminal) to
stop — the improved fighter is shipped to `models/fighter2.pt` (previous one
backed up to `fighter2_prev.pt`). Useful flags: `--resume <ckpt>`, `--lr`,
`--steps`, `--no-ship`.

**Fighting the real Theobald bot.** The bot needs the **HeroBot** Fabric mod +
Theobald's practice map, which target MC 1.21.x — a mod for one MC version can't
load in another, and one game instance runs one version. So to fight the literal
bot you need HeroBot + the map built for **this project's MC version (26.1.2)**;
if that isn't available, the native `theobald` opponent (stage 5 of
`train_all.sh`) already trains against a faithful reimplementation, and
`pilot_train.py` still fine-tunes against any real player. Once versions line
up: launch a client with **both** mods, load Theobald's map, spawn the bot at a
difficulty, run `pilot_train.py`, and press **G** — the pilot targets the bot
like any player, no extra config.

## Ports (all localhost-only)

| Port | Who listens | Used for |
|---|---|---|
| 36565 | mod (training server) | lock-step training bridge |
| 36566 | `pilot.py` / `pilot_train.py` | pilot inference / in-game fine-tuning |
| 8080 | dashboard | episode replays |
| 25565 | training server | normal Minecraft joins (e.g. to spectate) |
