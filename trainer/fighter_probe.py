"""Scripted verification of the stage-6 fighter environment (protocol v3).

Run with the server up and the bridge free:  ../.venv/bin/python fighter_probe.py

Phase A — movement axes: telemetry displacement must match each move/strafe
input relative to the agent's facing (W ahead, S behind, A left, D right).
Phase B — humanized opponent: with the agent standing passively in reach,
hits arrive noticeably slower than the perfect bot's cooldown cadence
(reaction delay), and juggling pressure (constant sprint knockback) must
suppress its output vs. the perfect bot (hitstun/airborne disruption).
"""

import numpy as np

from drlagent.protocol import BridgeConnection
from drlagent.vec_env import INFO_HIT_LANDED, INFO_HIT_TAKEN

CURR = {"horizontal_prob": 1.0, "max_elev": 0.0, "dist_min": 3.0,
        "dist_max": 3.0, "yaw_range": 0.0, "episode_ticks": 400,
        "lock_ticks": 3, "opponent": "human",
        "opp_reaction_min": 3, "opp_reaction_max": 7, "opp_fight_prob": 0.0}


def step(conn, move=0, strafe=0, attack=0, jump=0, sprint=0, reset=0, dyaw=None):
    n = conn.hello.arenas
    z = np.zeros(n, np.float32)
    if np.isscalar(attack):
        attack = np.full(n, attack, np.uint8)
    conn.send_actions(z if dyaw is None else dyaw.astype(np.float32), z,
                      attack.astype(np.uint8),
                      np.full(n, move, np.uint8), np.full(n, jump, np.uint8),
                      np.full(n, sprint, np.uint8),
                      reset=np.full(n, reset, np.uint8),
                      strafe=np.full(n, strafe, np.uint8))
    return conn.recv_obs()


def chase_inputs(obs):
    """Steer toward the opponent from telemetry; attack at full charge in
    reach — the scripted sprint-juggler that drives the disruption test."""
    tel = obs.telemetry
    dx = tel[:, 4] - tel[:, 0]
    dz = tel[:, 5] - tel[:, 1]
    target_yaw = np.degrees(np.arctan2(-dx, dz))
    dyaw = (target_yaw - tel[:, 3] + 180) % 360 - 180  # mod clamps to ±15
    dist = np.hypot(dx, dz)
    return dyaw, dist


def reconfigure(conn, **over):
    """Applied at the next tick; reseed force-resets every arena. The mod
    sends no reply for mid-session configs — alternation continues."""
    cur = dict(CURR, **over)
    conn.send_json({"msg": "config", "stage": "fighter", "seed": 99,
                    "curriculum": cur, "record_arena": 0, "reseed": True})


def displacement(conn, move, strafe, ticks=20):
    """Mean displacement direction (arena 0), in the agent's yaw frame."""
    obs = step(conn, reset=1)
    for _ in range(3):
        obs = step(conn)
    t0 = obs.telemetry[0]
    yaw = np.radians(t0[3])
    for _ in range(ticks):
        obs = step(conn, move=move, strafe=strafe)
    t1 = obs.telemetry[0]
    dx, dz = t1[0] - t0[0], t1[1] - t0[1]
    # forward basis (MC yaw 0 = +Z): f = (-sin, cos); left = (cos, sin)... derive:
    fx, fz = -np.sin(yaw), np.cos(yaw)
    lx, lz = np.cos(yaw), np.sin(yaw)  # 90° left of f: rotate f by +90° about Y
    return dx * fx + dz * fz, dx * lx + dz * lz  # (ahead, leftward)


def taken_per_episode(conn, opponent, juggle, episodes=6):
    reconfigure(conn, opponent=opponent,
                opp_fight_prob=1.0 if opponent == "fight" else 0.0)
    n = conn.hello.arenas
    taken = np.zeros(n)
    hits = np.zeros(n)
    done_eps, per_ep = 0, []
    obs = step(conn, reset=1)
    charge_i = conn.hello.scalars.index("cooldown")
    while done_eps < episodes:
        if juggle:
            charge = obs.scalars[:, charge_i]
            dyaw, dist = chase_inputs(obs)
            # swing only in reach: air swings reset the ticker and would
            # leave the charge empty by the time the target is reachable
            attack = ((charge >= 0.999) & (dist < 2.9)).astype(np.uint8)
            obs = step(conn, move=1, sprint=1, attack=attack, dyaw=dyaw)
        else:
            obs = step(conn)  # stand still, take it
        taken += (obs.infos & INFO_HIT_TAKEN) > 0
        hits += (obs.infos & INFO_HIT_LANDED) > 0
        for i in np.nonzero(obs.dones)[0]:
            per_ep.append((taken[i], hits[i]))
            taken[i] = hits[i] = 0
            done_eps += 1
    t = [p[0] for p in per_ep]
    h = [p[1] for p in per_ep]
    return float(np.mean(t)), float(np.mean(h))


def main():
    conn = BridgeConnection()
    print(f"connected: protocol {conn.hello.protocol}, {conn.hello.arenas} arenas")
    assert conn.hello.protocol == 3, "server jar predates protocol v3"
    conn.configure(stage="fighter", curriculum=CURR, seed=99)
    conn.recv_obs()  # initial obs — skipping it desyncs every later read by one

    print("\nPhase A — movement axes (displacement in the agent's yaw frame)")
    for name, mv, st, expect in [("W", 1, 0, "ahead"), ("S", 2, 0, "behind"),
                                 ("A", 0, 1, "left"), ("D", 0, 2, "right")]:
        ahead, left = displacement(conn, mv, st)
        print(f"  {name}: ahead {ahead:+.2f}  left {left:+.2f}")
        if expect == "ahead":
            assert ahead > 1.0 and abs(left) < ahead * 0.4, name
        elif expect == "behind":
            assert ahead < -1.0 and abs(left) < -ahead * 0.4, name
        elif expect == "left":
            assert left > 1.0 and abs(ahead) < left * 0.4, name
        else:
            assert left < -1.0 and abs(ahead) < -left * 0.4, name
    print("  PASS")

    print("\nPhase B — humanized opponent")
    t_fight, _ = taken_per_episode(conn, "fight", juggle=False)
    t_human, _ = taken_per_episode(conn, "human", juggle=False)
    print(f"  standing target: taken/ep fight={t_fight:.1f} human={t_human:.1f}")
    assert t_human < t_fight, "reaction delay should slow the humanized bot"
    assert t_human > 5, "humanized bot should still fight a passive target"

    tj_fight, hj_fight = taken_per_episode(conn, "fight", juggle=True)
    tj_human, hj_human = taken_per_episode(conn, "human", juggle=True)
    print(f"  under juggling: taken/ep fight={tj_fight:.1f} human={tj_human:.1f} "
          f"(hits landed {hj_fight:.1f}/{hj_human:.1f})")
    assert tj_human < tj_fight * 0.65, \
        "hitstun/airborne disruption should suppress the humanized bot's output"
    print("  PASS")

    print("\nALL PASSED")
    conn.close()


if __name__ == "__main__":
    main()
