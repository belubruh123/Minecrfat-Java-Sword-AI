"""Acceptance probe for the swing stage: frozen aim net steers, a scripted
attacker presses attack whenever the cooldown is full. Verifies hit rewards,
whiff penalties, info bits, opponent movement, and episode length.

Usage: python swing_probe.py ../runs/stage2_vertical_v2/best.pt
"""

import sys

import numpy as np
import torch

from drlagent.models import AimPolicy
from drlagent.vec_env import (INFO_HIT_LANDED, INFO_ON_TARGET, INFO_WHIFF,
                              MinecraftVecEnv)

CURRICULUM = {
    "horizontal_prob": 1.0, "max_elev": 0.0,
    "dist_min": 2.0, "dist_max": 6.0, "yaw_range": 90.0,
    "episode_ticks": 300, "lock_ticks": 3, "opponent": "strafe",
}
COOLDOWN_IDX = 6  # scalar order: speed, yaw_sin, yaw_cos, pitch, y, ground, cooldown, health, reach
STEPS = 1500


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> None:
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    env = MinecraftVecEnv(curriculum=CURRICULUM, stage="swing", seed=99,
                          downsample=3, frame_stack=4, record_dir=None)
    aim = AimPolicy(env.stack, env.h, env.w, env.n_scalars)
    aim.load_state_dict(ckpt["policy"])
    aim.eval()

    hit_rewards, whiff_rewards, other_rewards = [], [], []
    centroids = []
    ep_lens = []
    on_target_all = on_target_atk = atk_ticks = total_ticks = 0
    with torch.no_grad():
        for _ in range(STEPS):
            masks, scalars = env.observe()
            # swing only when charged AND the target blob is big (close) —
            # the same pixel cue the swing model is expected to learn
            area = masks[:, -1].sum(axis=(1, 2))
            attack = ((scalars[:, COOLDOWN_IDX] >= 0.999) & (area > 350)).astype(np.uint8)
            a, _, _, _ = aim.act(torch.from_numpy(masks).float(),
                                 torch.from_numpy(scalars), deterministic=True)
            a = a.numpy()
            rewards, dones, infos = env.step(a[:, 0], a[:, 1], attack=attack)
            on = (infos & INFO_ON_TARGET) > 0
            on_target_all += int(on.sum())
            on_target_atk += int((on & (attack > 0)).sum())
            atk_ticks += int(attack.sum())
            total_ticks += env.n
            for i in range(env.n):
                if infos[i] & INFO_HIT_LANDED:
                    hit_rewards.append(rewards[i])
                elif infos[i] & INFO_WHIFF:
                    whiff_rewards.append(rewards[i])
                else:
                    other_rewards.append(rewards[i])
            ys, xs = np.nonzero(masks[0, -1])
            if len(xs):
                centroids.append(xs.mean())
    ep_lens = [s["length"] for s in env.drain_stats()]

    hits, whiffs = len(hit_rewards), len(whiff_rewards)
    print(f"diag: on-target {on_target_all/total_ticks:.0%} of all ticks, "
          f"{on_target_atk/max(atk_ticks,1):.0%} of attack ticks ({atk_ticks} attacks)")
    ok = True
    ok &= check("hits land", hits > 20, f"{hits} hits")
    ok &= check("hit reward ~ charge", hits > 0 and 0.5 <= float(np.mean(hit_rewards)) <= 1.0,
                f"mean {np.mean(hit_rewards):.2f}" if hits else "none")
    ok &= check("whiffs penalized", whiffs > 0 and abs(float(np.mean(whiff_rewards)) + 0.3) < 1e-4,
                f"{whiffs} whiffs, mean {np.mean(whiff_rewards):.2f}" if whiffs else "none")
    ok &= check("no-attack ticks pay 0", not other_rewards or abs(float(np.mean(other_rewards))) < 1e-5,
                f"mean {np.mean(other_rewards):.4f}")
    ok &= check("opponent strafes (blob moves)", len(centroids) > 50 and float(np.std(centroids)) > 2.0,
                f"centroid std {np.std(centroids):.1f}px")
    ok &= check("episodes run full length", bool(ep_lens) and max(ep_lens) >= 295,
                f"lens {sorted(set(ep_lens))[:5]}..." if ep_lens else "none finished")
    env.close()
    print("ALL PASSED" if ok else "PROBE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
