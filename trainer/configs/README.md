# Training configs

## Train from zero — one command (recommended)

From the repo root (the server is started/rebooted for you):

```bash
scripts/train_all.sh            # full budgets — best model (~12h on CPU)
scripts/train_all.sh quick      # decent model (~4-5h)
scripts/train_all.sh smoke      # tiny — just proves the pipeline runs end-to-end
```

Or set your own budgets (steps; 1 step = 1 tick across 8 arenas):

```bash
AIM_STEPS=1000000 VERT_STEPS=1500000 COMBO_STEPS=1500000 FIGHT_STEPS=250000 \
    scripts/train_all.sh
```

It runs all four stages below, reboots the server safely between segments
(long single boots corrupt hit-registration), warm-starts each stage from the
previous, and **ships `models/aim.pt` + `models/fighter2.pt`** at the end (your
old models are backed up to `*_prev.pt`). The pilot then auto-uses them — press
**G** in game (see `SETUP.md`). Interrupted? Re-run the same command; each stage
resumes from its own `runs/fz_*/latest.pt`. Delete a `runs/fz_*` dir to redo a
stage. The fight stage is short and stops at its best checkpoint on purpose —
on the high-variance mortal task the policy eventually entropy-drifts, so
best-return checkpointing captures the peak (that is what gets shipped).

## Train from zero — manual (the four stages the script runs)

Run these in order from `trainer/` (server up via `scripts/start_server.sh`):

| # | Config | Trains | Notes |
|---|---|---|---|
| 1 | `stage1_aim.yaml` | aim from scratch | smooth-aim doctrine (flick-or-hold, ≥5° moves, 7.5°/tick cap) is enforced by the environment, so a fresh aim learns it natively |
| 2 | `stage2_vertical.yaml` | vertical aim | `--resume ../runs/stage1_aim/final.pt`; keeps 30% eye-level rehearsal so horizontal aim is not forgotten |
| 3 | `stage7a_comboschool.yaml` | the fighter (combo school) | copy your aim into `models/aim.pt` first; **comment out `init_checkpoint`** to start the fighter from scratch (it exists to resume our lineage) |
| 4 | `stage7b_chain.yaml` | the fighter (mortal duels) | warm-starts from your school run automatically |

Ship the result: `cp ../runs/stage7b_chain/final.pt ../models/fighter2.pt` —
the pilot picks it up on restart.

The `fighter2` stages own the attack button (72-way action head), so no
separate swing/move/combo models are needed — that is why the from-zero
path is only four configs.

## Our lineage / special-purpose

- `aim_smooth_v4.yaml` — fine-tune that converted the original snap-aim to
  the smooth doctrine and the half-speed camera. Only needed if you have an
  old-generation `aim.pt`; a from-zero aim learns the doctrine in stage 1.

## `archive/`

Superseded generations (swing/move/combo/fighter stages 3–6, earlier aim
passes). Their outputs ship in `models/` and nothing in the live curriculum
depends on them; kept for reproducibility of the historical models.
