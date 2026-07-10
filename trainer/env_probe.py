"""Environment verification probe (task-2 acceptance test).

Connects to the mod and runs scripted checks against a live server:
  A. physics sanity  - no-op actions: agent stays on the floor, speed ~0
  B. spin sweep      - constant +dyaw: the white blob appears and its centroid
                       sweeps across the screen
  C. aim convergence - P-controller on the blob centroid centers the target
                       and the on-target info bit + lock episode-end fire
  D. movement        - holding W produces forward speed (sprint ~0.28 b/t
                       once ramped; flags double-ticking if ~2x)
  E. attack          - walk at the target while aiming and clicking: a hit
                       lands and last_reach becomes > 0

Run the dev server first: DRLAGENT_TRAIN=1 ./gradlew runServer
"""

import numpy as np

from drlagent.protocol import BridgeConnection

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{results[-1][1]}] {name}: {detail}")


def centroid_x(mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(mask)
    return None if xs.size == 0 else float(xs.mean())


def step_all(conn, dyaw=0.0, dpitch=0.0, attack=0, forward=0, reset=0):
    n = conn.hello.arenas
    conn.send_actions(
        np.full(n, dyaw, np.float32), np.full(n, dpitch, np.float32),
        np.full(n, attack, np.uint8), np.full(n, forward, np.uint8),
        np.full(n, reset, np.uint8))
    return conn.recv_obs()


def reset_episode(conn):
    return step_all(conn, reset=1)


def main() -> None:
    conn = BridgeConnection()
    conn.configure(stage="aim", curriculum={
        "horizontal_prob": 1.0, "dist_min": 4, "dist_max": 6,
        "yaw_range": 180, "episode_ticks": 400, "opponent": "static",
    }, seed=42)
    obs = conn.recv_obs()  # initial observation
    n = conn.hello.arenas
    check("handshake+initial obs", obs.masks.shape == (n, 240, 426),
          f"masks {obs.masks.shape}, scalars {obs.scalars.shape}")

    # --- A: physics sanity ---------------------------------------------------
    ys, speeds, grounded = [], [], []
    for _ in range(20):
        obs = step_all(conn)
        ys.append(obs.scalars[0][4])
        speeds.append(obs.scalars[0][0])
        grounded.append(obs.scalars[0][5])
    check("A physics: stays on floor", max(abs(y) for y in ys) < 0.01,
          f"y range [{min(ys):.4f},{max(ys):.4f}]")
    check("A physics: no drift", max(speeds) < 0.001, f"max speed {max(speeds):.5f}")
    check("A physics: on_ground flag", grounded[-1] == 1.0, f"on_ground={grounded[-1]}")

    # --- B: spin sweep -------------------------------------------------------
    obs = reset_episode(conn)
    cxs, yaw_at_seen = [], []
    for _ in range(80):  # 80 * 5 deg = 400 deg, > full turn
        obs = step_all(conn, dyaw=5.0)
        cx = centroid_x(obs.masks[0])
        if cx is not None and not obs.dones[0]:
            cxs.append(cx)
            yaw = np.degrees(np.arctan2(obs.scalars[0][1], obs.scalars[0][2]))
            yaw_at_seen.append(yaw)
    seen = len(cxs)
    sweep = (max(cxs) - min(cxs)) if seen >= 3 else 0
    check("B spin: target visible during sweep", seen >= 3,
          f"visible on {seen}/80 ticks")
    check("B spin: centroid sweeps horizontally", sweep > 100,
          f"centroid x range {sweep:.0f}px over {seen} sightings")

    # --- C: aim convergence via P-controller on the mask --------------------
    locks = 0
    on_target_ticks = 0
    tick_count = 0
    obs = reset_episode(conn)
    for _ in range(600):
        cx = centroid_x(obs.masks[0])
        if cx is None:
            dyaw = 8.0  # search
        else:
            dyaw = 0.05 * (cx - 426 / 2)  # P-control toward screen center
        obs = step_all(conn, dyaw=dyaw)
        tick_count += 1
        if obs.infos[0] & 1:
            on_target_ticks += 1
        if obs.dones[0]:
            locks += 1
            if locks >= 3:
                break
    check("C aim: on-target bit fires", on_target_ticks > 0,
          f"{on_target_ticks} on-target ticks")
    check("C aim: P-controller achieves locks", locks >= 3,
          f"{locks} lock episodes in {tick_count} ticks")

    # --- D: movement ---------------------------------------------------------
    obs = reset_episode(conn)
    speeds = []
    for _ in range(40):
        obs = step_all(conn, forward=1)
        speeds.append(float(obs.scalars[0][0]))
    top = max(speeds)
    check("D move: W produces speed", top > 0.15, f"top speed {top:.3f} b/t")
    check("D move: not double-ticked", top < 0.45,
          f"top speed {top:.3f} (sprint ~0.28 expected, ~0.56 would mean double tick)")

    # --- E: attack -----------------------------------------------------------
    # Disable aim-lock episode ends so the approach isn't interrupted.
    conn.send_json({"msg": "config", "stage": "aim", "seed": 7, "curriculum": {
        "horizontal_prob": 1.0, "dist_min": 4, "dist_max": 6,
        "yaw_range": 180, "episode_ticks": 100000, "lock_ticks": 100000,
        "opponent": "static"}})
    hit, reach = False, 0.0
    for episode in range(5):
        obs = reset_episode(conn)
        for _ in range(400):
            cx = centroid_x(obs.masks[0])
            dyaw = 8.0 if cx is None else 0.05 * (cx - 213)
            aimed = cx is not None and abs(cx - 213) < 40
            blob = int(obs.masks[0].sum())
            close = blob > 4200  # ~3 blocks away at 426x240, 70 deg FOV
            obs = step_all(conn, dyaw=dyaw, forward=0 if close else 1,
                           attack=1 if aimed and blob > 2500 else 0)
            if obs.infos[0] & 2:
                hit = True
                reach = float(obs.scalars[0][8])
                break
        if hit:
            break
    check("E attack: hit lands", hit, f"hit={hit}")
    check("E attack: reach measured", 0.5 < reach <= 3.5, f"last_reach={reach:.2f}")

    conn.close()
    failed = [r for r in results if r[1] == FAIL]
    print(f"\n{'ALL PASSED' if not failed else f'{len(failed)} FAILED'} "
          f"({len(results)} checks)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
