"""Evaluate a checkpoint with a deterministic policy.

Usage: python eval.py runs/stage1_aim/latest.pt --episodes 100 [--curriculum configs/stage1_aim.yaml]
Swing checkpoints need the frozen aim net: --aim runs/stage2_vertical_v2/best.pt
Move checkpoints need both: --aim ... --swing runs/stage3_swing/latest.pt
Stop the trainer first — the bridge holds one session at a time.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from drlagent.models import (AimPolicy, ComboPolicy, Fighter2Policy,
                             FighterPolicy, MovePolicy, SwingPolicy)
from drlagent.vec_env import MinecraftVecEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--curriculum", default=None,
                    help="yaml config supplying the eval curriculum")
    ap.add_argument("--aim", default=None,
                    help="aim checkpoint that steers during a swing/move eval")
    ap.add_argument("--swing", default=None,
                    help="swing checkpoint that attacks during a move eval")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of argmax (aim stays deterministic)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--record", default=None,
                    help="episode record dir (for the dashboard replay)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stack, h, w, n_scalars = ckpt["obs_shape"]
    ckpt_stage = ckpt.get("stage", "aim")

    curriculum = {"horizontal_prob": 1.0, "yaw_range": 90.0}
    stage = ckpt_stage
    if args.curriculum:
        cfg = yaml.safe_load(Path(args.curriculum).read_text())
        curriculum = cfg["curriculum"]
        stage = cfg["stage"]

    env = MinecraftVecEnv(curriculum=curriculum, stage=stage, seed=args.seed,
                          downsample=426 // w, frame_stack=stack,
                          record_dir=args.record)
    assert (env.h, env.w, env.n_scalars) == (h, w, n_scalars), \
        f"env obs {(env.h, env.w, env.n_scalars)} != checkpoint {(h, w, n_scalars)}"

    aim = None
    swing = None
    if ckpt_stage in ("swing", "move", "combo", "fighter", "fighter2"):
        if not args.aim:
            raise SystemExit(f"{ckpt_stage} checkpoint: pass --aim <aim checkpoint>")
        aim = AimPolicy(stack, h, w, n_scalars)
        aim.load_state_dict(torch.load(args.aim, map_location="cpu",
                                       weights_only=False)["policy"])
        aim.eval()
    if ckpt_stage == "swing":
        policy = SwingPolicy(stack, h, w, n_scalars)
    elif ckpt_stage == "fighter2":
        policy = Fighter2Policy(stack, h, w, n_scalars)
    elif ckpt_stage in ("move", "combo", "fighter"):
        if not args.swing:
            raise SystemExit(f"{ckpt_stage} checkpoint: pass --swing <swing checkpoint>")
        swing = SwingPolicy(stack, h, w, n_scalars)
        swing.load_state_dict(torch.load(args.swing, map_location="cpu",
                                         weights_only=False)["policy"])
        swing.eval()
        policy = {"move": MovePolicy, "combo": ComboPolicy,
                  "fighter": FighterPolicy}[ckpt_stage](stack, h, w, n_scalars)
    else:
        policy = AimPolicy(stack, h, w, n_scalars)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    stats: list[dict] = []
    with torch.no_grad():
        while len(stats) < args.episodes:
            masks, scalars = env.observe()
            tm = torch.from_numpy(masks).float()
            ts = torch.from_numpy(scalars)
            action, _, _, _ = policy.act(tm, ts,
                                         deterministic=not args.stochastic)
            if ckpt_stage == "move":
                aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                # the swing head is sampled: that is its trained behavior
                swing_a, _, _, _ = swing.act(tm, ts)
                env.step(aim_a[:, 0], aim_a[:, 1],
                         attack=swing_a.numpy().astype(np.uint8),
                         forward=action.numpy().astype(np.uint8))
            elif ckpt_stage == "combo":
                aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                swing_a, _, _, _ = swing.act(tm, ts)
                a = action.numpy().astype(np.int64)
                env.step(aim_a[:, 0], aim_a[:, 1],
                         attack=swing_a.numpy().astype(np.uint8),
                         forward=(a & 1).astype(np.uint8),
                         jump=((a >> 1) & 1).astype(np.uint8),
                         sprint=((a >> 2) & 1).astype(np.uint8))
            elif ckpt_stage == "fighter2":
                aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                a = action.numpy().astype(np.int64)
                move, strafe, jump, sprint, attack = Fighter2Policy.decode(a)
                env.step(aim_a[:, 0], aim_a[:, 1],
                         attack=attack.astype(np.uint8),
                         move=move.astype(np.uint8),
                         strafe=strafe.astype(np.uint8),
                         jump=jump.astype(np.uint8),
                         sprint=sprint.astype(np.uint8))
            elif ckpt_stage == "fighter":
                aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                swing_a, _, _, _ = swing.act(tm, ts)
                a = action.numpy().astype(np.int64)
                move, strafe, jump, sprint = FighterPolicy.decode(a)
                env.step(aim_a[:, 0], aim_a[:, 1],
                         attack=swing_a.numpy().astype(np.uint8),
                         move=move.astype(np.uint8),
                         strafe=strafe.astype(np.uint8),
                         jump=jump.astype(np.uint8),
                         sprint=sprint.astype(np.uint8))
            elif aim is not None:
                aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                env.step(aim_a[:, 0], aim_a[:, 1],
                         attack=action.numpy().astype(np.uint8))
            else:
                a = action.numpy()
                env.step(a[:, 0], a[:, 1])
            stats.extend(env.drain_stats())

    stats = stats[: args.episodes]
    rets = [s["return"] for s in stats]
    print(f"checkpoint: {args.checkpoint} (trained {ckpt['step']} steps)")
    print(f"episodes:   {len(stats)}   mean length {np.mean([s['length'] for s in stats]):.0f} ticks")
    if ckpt_stage in ("swing", "move", "combo", "fighter", "fighter2"):
        hits = [s["hits"] for s in stats]
        whiffs = [s["whiffs"] for s in stats]
        swings = sum(hits) + sum(whiffs)
        print(f"hits/ep:    mean {np.mean(hits):.1f}   whiffs/ep: mean {np.mean(whiffs):.1f}")
        print(f"hit rate:   {sum(hits) / max(swings, 1):.1%} of {swings} swings")
        if ckpt_stage in ("move", "combo", "fighter", "fighter2"):
            taken = [s["hits_taken"] for s in stats]
            print(f"hits taken: mean {np.mean(taken):.1f}/ep "
                  f"(damage ratio {sum(hits) / max(sum(taken), 1):.2f})")
        if ckpt_stage in ("combo", "fighter", "fighter2"):
            crits = [s["crits"] for s in stats]
            sprints = [s["sprint_hits"] for s in stats]
            chains = [s.get("chain_hits", 0) for s in stats]
            print(f"crits:      mean {np.mean(crits):.1f}/ep   "
                  f"sprint hits: mean {np.mean(sprints):.1f}/ep")
            print(f"combo hits: mean {np.mean(chains):.1f}/ep "
                  f"(hits landed <=30 ticks after the previous, nothing taken between)")
    else:
        succ = [s["success"] for s in stats]
        lens = [s["length"] for s in stats if s["success"]]
        print(f"success:    {np.mean(succ):.1%}")
        print(f"time-to-lock (successful): median {np.median(lens) if lens else float('nan'):.0f} "
              f"mean {np.mean(lens) if lens else float('nan'):.0f} ticks")
    print(f"return:     mean {np.mean(rets):.2f}")
    env.close()


if __name__ == "__main__":
    main()
