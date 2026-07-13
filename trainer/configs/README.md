# Training configs

## Train from zero (no pretrained models needed)

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
