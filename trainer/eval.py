"""Evaluate a checkpoint with a deterministic policy.

Usage: python eval.py runs/stage1_aim/latest.pt --episodes 100 [--curriculum configs/stage1_aim.yaml]
Stop the trainer first — the bridge holds one session at a time.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from drlagent.models import AimPolicy
from drlagent.vec_env import MinecraftVecEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--curriculum", default=None,
                    help="yaml config supplying the eval curriculum")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stack, h, w, n_scalars = ckpt["obs_shape"]

    curriculum = {"horizontal_prob": 1.0, "yaw_range": 90.0}
    stage = "aim"
    if args.curriculum:
        cfg = yaml.safe_load(Path(args.curriculum).read_text())
        curriculum = cfg["curriculum"]
        stage = cfg["stage"]

    env = MinecraftVecEnv(curriculum=curriculum, stage=stage, seed=args.seed,
                          downsample=426 // w, frame_stack=stack,
                          record_dir=None)
    assert (env.h, env.w, env.n_scalars) == (h, w, n_scalars), \
        f"env obs {(env.h, env.w, env.n_scalars)} != checkpoint {(h, w, n_scalars)}"

    policy = AimPolicy(stack, h, w, n_scalars)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    stats: list[dict] = []
    with torch.no_grad():
        while len(stats) < args.episodes:
            masks, scalars = env.observe()
            action, _, _, _ = policy.act(torch.from_numpy(masks).float(),
                                         torch.from_numpy(scalars),
                                         deterministic=True)
            a = action.numpy()
            env.step(a[:, 0], a[:, 1])
            stats.extend(env.drain_stats())

    stats = stats[: args.episodes]
    succ = [s["success"] for s in stats]
    lens = [s["length"] for s in stats if s["success"]]
    rets = [s["return"] for s in stats]
    print(f"checkpoint: {args.checkpoint} (trained {ckpt['step']} steps)")
    print(f"episodes:   {len(stats)}")
    print(f"success:    {np.mean(succ):.1%}")
    print(f"time-to-lock (successful): median {np.median(lens) if lens else float('nan'):.0f} "
          f"mean {np.mean(lens) if lens else float('nan'):.0f} ticks")
    print(f"return:     mean {np.mean(rets):.2f}")
    env.close()


if __name__ == "__main__":
    main()
