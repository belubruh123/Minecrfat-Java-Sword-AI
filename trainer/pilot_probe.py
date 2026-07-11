"""End-to-end smoke test for trainer/pilot.py: starts the server, connects a
fake client speaking the mod's exact framing, and checks the actions."""

import json
import socket
import struct
import subprocess
import sys
import time

import numpy as np

W, H = 426, 240
TYPE_JSON, TYPE_ACTION, TYPE_OBS = 0, 1, 2
PORT = 36567


def send_frame(f, ftype, payload):
    f.write(struct.pack(">IB", len(payload) + 1, ftype))
    f.write(payload)
    f.flush()


def read_frame(f, expected):
    length, ftype = struct.unpack(">IB", f.read(5))
    payload = f.read(length - 1)
    assert ftype == expected, f"expected type {expected}, got {ftype}"
    return payload


def make_obs(cx_frac):
    """Opponent-ish blob centered at cx_frac of the width."""
    mask = np.zeros((H, W), np.uint8)
    cx = int(W * cx_frac)
    mask[95:150, max(0, cx - 14):min(W, cx + 14)] = 1
    scalars = np.array([0.1, 0.0, 1.0, 0.05, 0.0, 1.0, 1.0, 1.0, 2.5], ">f4")
    return np.packbits(mask.flatten()).tobytes() + scalars.tobytes()


proc = subprocess.Popen(
    [sys.executable, "-u", "pilot.py", "--port", str(PORT)],
    cwd="/root/DRL-Minecraft-MOD/trainer",
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            print("server died:\n" + "".join(lines))
            sys.exit(1)
        lines.append(line)
        print("server:", line, end="")
        if "listening" in line:
            break

    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    f = s.makefile("rwb")
    send_frame(f, TYPE_JSON, json.dumps(
        {"msg": "pilot_hello", "protocol": 1, "width": W, "height": H}).encode())
    reply = json.loads(read_frame(f, TYPE_JSON))
    assert reply["msg"] == "ready", reply
    print("handshake OK")

    def run(cx, n):
        acts = []
        for _ in range(n):
            send_frame(f, TYPE_OBS, make_obs(cx))
            dyaw, dpitch, attack, forward = struct.unpack(
                ">ffBB", read_frame(f, TYPE_ACTION))
            assert -15.001 <= dyaw <= 15.001 and -15.001 <= dpitch <= 15.001
            assert attack in (0, 1) and forward in (0, 1)
            acts.append((dyaw, dpitch, attack, forward))
        return acts

    t0 = time.perf_counter()
    right = run(0.85, 30)   # blob far right of center
    left = run(0.15, 30)    # blob far left of center
    center = run(0.5, 30)
    ms = (time.perf_counter() - t0) * 1000 / 90

    r_yaw = np.mean([a[0] for a in right])
    l_yaw = np.mean([a[0] for a in left])
    c_yaw = np.mean([a[0] for a in center])
    att = np.mean([a[2] for a in right + left + center])
    fwd = np.mean([a[3] for a in right + left + center])
    print(f"mean dyaw: right-blob {r_yaw:+.2f}  left-blob {l_yaw:+.2f}  "
          f"center {c_yaw:+.2f} deg/tick")
    print(f"attack rate {att:.0%}  forward rate {fwd:.0%}  {ms:.1f} ms/tick")
    assert r_yaw > 1.0, "aim should turn right toward a right-side blob"
    assert l_yaw < -1.0, "aim should turn left toward a left-side blob"
    assert abs(c_yaw) < abs(r_yaw) and abs(c_yaw) < abs(l_yaw), \
        "centered blob should need less correction"
    print("ALL PASSED")
finally:
    proc.terminate()
