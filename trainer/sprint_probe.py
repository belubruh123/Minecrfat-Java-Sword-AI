"""Measure fake-player ground movement: walk vs sprint speed.

Vanilla reference: walk 4.317 m/s = 0.216 b/t, sprint 5.612 m/s = 0.281 b/t.
Run with the trainer stopped (the bridge holds one session).
"""

import numpy as np

from drlagent.vec_env import MinecraftVecEnv

CUR = {"horizontal_prob": 1.0, "max_elev": 0.0, "dist_min": 20.0, "dist_max": 20.0,
       "yaw_range": 0.0, "episode_ticks": 10_000, "lock_ticks": 100_000,
       "opponent": "static"}
SPEED_IDX = 0  # raw scalar; env normalizes by 1/0.3, so un-normalize below


def measure(env, ticks: int = 50) -> list[float]:
    z = np.zeros(env.n, np.float32)
    fwd = np.ones(env.n, np.uint8)
    speeds = []
    for _ in range(ticks):
        _, scal = env.observe()
        speeds.append(float(scal[0][SPEED_IDX]) * 0.3)  # undo SCALAR_SCALE
        env.step(z, z, forward=fwd)
    return speeds


def main() -> None:
    env = MinecraftVecEnv(curriculum=CUR, stage="aim", seed=5, downsample=3,
                          frame_stack=1, record_dir=None)
    speeds = measure(env)
    plateau = float(np.mean(speeds[-15:]))
    print("speed by tick:", " ".join(f"{s:.3f}" for s in speeds[:20]), "...")
    print(f"plateau speed: {plateau:.3f} b/t "
          f"(walk=0.216, sprint=0.281)")
    if plateau > 0.26:
        print("SPRINT ENGAGED")
    elif plateau > 0.19:
        print("WALKING ONLY — sprint flag not taking effect")
    else:
        print("UNEXPECTEDLY SLOW — inputs or physics broken")
    env.close()


if __name__ == "__main__":
    main()
