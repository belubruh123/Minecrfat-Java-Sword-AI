"""Verifies the combo-stage mechanics before training: sprint-hits must flag
INFO_SPRINT_HIT (bit 64), jump-crits must flag INFO_CRIT (bit 32), and the two
must be mutually exclusive — mirroring 26.1's Player.attack.

Phase A: hold W+sprint, attack at full charge -> expect sprint hits, no crits.
Phase B: never sprint, jump and attack only while falling -> expect crits.

The frozen aim model steers (as in training); attacks are scripted.
Run from trainer/ with the server up: ../.venv/bin/python combo_probe.py
"""

import numpy as np
import torch

from drlagent.models import AimPolicy
from drlagent.vec_env import (INFO_CRIT, INFO_HIT_LANDED, INFO_SPRINT_HIT,
                              MinecraftVecEnv)

CURRICULUM = {
    "horizontal_prob": 1.0, "max_elev": 0.0, "dist_min": 2.5, "dist_max": 2.5,
    "yaw_range": 0.0, "episode_ticks": 300, "lock_ticks": 3, "opponent": "strafe",
}
TICKS = 1800  # per phase, all arenas


def run_phase(env, aim, mode):
    n = env.n
    hits = crits = sprints = 0
    prev_y = np.zeros(n, np.float32)
    jump_cd = np.zeros(n, np.int32)
    for _ in range(TICKS):
        masks, scalars = env.observe()
        tm = torch.from_numpy(masks).float()
        ts = torch.from_numpy(scalars)
        with torch.no_grad():
            aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
        aim_a = aim_a.numpy()

        area = masks[:, -1].sum(axis=(1, 2))
        charge = scalars[:, 6]
        on_ground = scalars[:, 5] > 0.5
        y = scalars[:, 4]

        if mode == "sprint":
            forward = np.ones(n, np.uint8)
            sprint = np.ones(n, np.uint8)
            jump = np.zeros(n, np.uint8)
            attack = ((charge >= 0.999) & (area > 350)).astype(np.uint8)
        else:  # crit: stand near, jump, swing on the way down
            forward = (area < 250).astype(np.uint8)  # close only when far
            sprint = np.zeros(n, np.uint8)
            jump_cd -= 1
            jump = (on_ground & (area > 250) & (jump_cd <= 0)).astype(np.uint8)
            jump_cd[jump > 0] = 25
            falling = ~on_ground & (y < prev_y - 1e-4)
            attack = ((charge >= 0.999) & falling & (area > 200)).astype(np.uint8)
        prev_y = y.copy()

        _, _, infos = env.step(aim_a[:, 0], aim_a[:, 1], attack=attack,
                               forward=forward, jump=jump, sprint=sprint)
        hits += int(((infos & INFO_HIT_LANDED) > 0).sum())
        crits += int(((infos & INFO_CRIT) > 0).sum())
        sprints += int(((infos & INFO_SPRINT_HIT) > 0).sum())
    env.drain_stats()
    return hits, crits, sprints


def main():
    torch.set_num_threads(2)
    env = MinecraftVecEnv(curriculum=CURRICULUM, stage="combo", seed=99,
                          downsample=3, frame_stack=4, record_dir=None)
    ckpt = torch.load("../models/aim.pt", map_location="cpu", weights_only=False)
    aim = AimPolicy(*ckpt["obs_shape"])
    aim.load_state_dict(ckpt["policy"])
    aim.eval()

    ok = True
    h, c, s = run_phase(env, aim, "sprint")
    print(f"phase A (sprint): hits={h} crits={c} sprint_hits={s}")
    if not (s >= 5 and c == 0 and s >= 0.5 * max(h, 1)):
        print("  FAIL: expected mostly sprint-flagged hits and zero crits")
        ok = False

    h, c, s = run_phase(env, aim, "crit")
    print(f"phase B (crit):   hits={h} crits={c} sprint_hits={s}")
    if not (c >= 3 and s == 0):
        print("  FAIL: expected several crits and zero sprint hits")
        ok = False

    env.close()
    print("ALL PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
