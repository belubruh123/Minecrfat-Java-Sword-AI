"""Knockback probe: land controlled hits on the strafe opponent and print its
position for the following ticks. Compare with [kbdbg] server-log lines."""
import sys

sys.path.insert(0, "/root/DRL-Minecraft-MOD/trainer")
import numpy as np
from drlagent.protocol import BridgeConnection
from drlagent.vec_env import INFO_HIT_LANDED
from fighter_probe import CURR, step, chase_inputs

OPPONENT = sys.argv[1] if len(sys.argv) > 1 else "strafe"
conn = BridgeConnection()
conn.configure(stage="fighter",
               curriculum=dict(CURR, opponent=OPPONENT), seed=123)
conn.recv_obs()  # initial obs — skipping it desyncs every later read by one
charge_i = conn.hello.scalars.index("cooldown")
obs = step(conn, reset=1)
hits, track = 0, 0
for t in range(400):
    charge = obs.scalars[:, charge_i]
    dyaw, dist = chase_inputs(obs)
    attack = ((charge >= 0.999) & (dist < 2.9)).astype(np.uint8)
    obs = step(conn, move=1, sprint=1, attack=attack, dyaw=dyaw)
    tel = obs.telemetry[0]
    if obs.infos[0] & INFO_HIT_LANDED:
        hits += 1
        track = 8
        print(f"tick {t}: HIT #{hits}  opp y={tel[6]:+.3f} "
              f"xz=({tel[4]:+.2f},{tel[5]:+.2f}) dist={dist[0]:.2f}")
    elif track > 0:
        track -= 1
        print(f"        +{8 - track}: opp y={tel[6]:+.3f} "
              f"xz=({tel[4]:+.2f},{tel[5]:+.2f}) dist={dist[0]:.2f}")
    if hits >= 3 and track == 0:
        break
print(f"total hits arena0: {hits}")
