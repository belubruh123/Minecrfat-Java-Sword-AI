"""Pilot inference server: lets the trained policies drive the user's own
character via the mod's client pilot mode.

Usage (from trainer/):
    python pilot.py [--aim CKPT] [--swing CKPT] [--move CKPT] [--port 36566]

Start this first, then press the pilot toggle key (default G) in game. The
client mod connects, streams one observation per client tick (mask + scalars,
PROTOCOL.md order), and gets back one action. Aim runs deterministic; swing
and move are sampled — their trained, canonical mode. Omitting --swing/--move
(or pointing at a missing file) disables that head: attack/forward stay 0, so
staged testing (aim-only, aim+swing) works before stage 4 finishes.
"""

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch

from drlagent.models import AimPolicy, MovePolicy, SwingPolicy
from drlagent.vec_env import MAX_TURN_DEG, MinecraftVecEnv

TYPE_JSON, TYPE_ACTION, TYPE_OBS = 0, 1, 2
MODELS = Path(__file__).resolve().parent.parent / "models"


def load_policy(cls, path: str | None, name: str):
    if not path:
        print(f"{name}: disabled (no checkpoint)")
        return None, None
    p = Path(path)
    if not p.exists():
        print(f"{name}: disabled ({p} does not exist)")
        return None, None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    net = cls(*ckpt["obs_shape"])
    net.load_state_dict(ckpt["policy"])
    net.eval()
    net.requires_grad_(False)
    print(f"{name}: {p} ({ckpt['step']} steps, stage {ckpt.get('stage', '?')})")
    return net, ckpt["obs_shape"]


def read_frame(f, expected_type):
    hdr = f.read(5)
    if len(hdr) < 5:
        raise ConnectionError("client closed")
    length, ftype = struct.unpack(">IB", hdr)
    payload = f.read(length - 1)
    if len(payload) < length - 1:
        raise ConnectionError("client closed mid-frame")
    if ftype != expected_type:
        raise ConnectionError(f"expected frame type {expected_type}, got {ftype}")
    return payload


def send_frame(f, ftype, payload):
    f.write(struct.pack(">IB", len(payload) + 1, ftype))
    f.write(payload)
    f.flush()


def serve_connection(conn, aim, swing, move, shape):
    stack, h, w, n_scalars = shape
    f = conn.makefile("rwb")
    hello = json.loads(read_frame(f, TYPE_JSON))
    width, height = hello["width"], hello["height"]
    ds = height // h
    if width // ds != w or height // ds != h:
        raise ConnectionError(f"obs {width}x{height} does not downsample to {w}x{h}")
    send_frame(f, TYPE_JSON, b'{"msg":"ready"}')
    print(f"client connected: {width}x{height} obs, downsample {ds} -> {w}x{h}")

    frames = np.zeros((stack, h, w), np.uint8)
    first = True
    mask_bytes = (width * height + 7) // 8
    n_ticks, n_attacks, n_forward, infer_ms = 0, 0, 0, 0.0

    while True:
        payload = read_frame(f, TYPE_OBS)
        t0 = time.perf_counter()
        mask = np.unpackbits(np.frombuffer(payload, np.uint8, mask_bytes))
        mask = mask.reshape(height, width)
        small = mask.reshape(h, ds, w, ds).max(axis=(1, 3))
        raw = np.frombuffer(payload, ">f4", n_scalars, mask_bytes).astype(np.float32)
        if first:
            frames[:] = small
            first = False
        else:
            frames[:-1] = frames[1:]
            frames[-1] = small
        scalars = MinecraftVecEnv._norm_scalars(raw)

        tm = torch.from_numpy(frames[None]).float()
        ts = torch.from_numpy(scalars[None])
        with torch.no_grad():
            aim_a, _, _, _ = aim.act(tm, ts, deterministic=True)
            attack = forward = 0
            if swing is not None:
                attack = int(swing.act(tm, ts)[0].item())
            if move is not None:
                forward = int(move.act(tm, ts)[0].item())
        dyaw = float(np.clip(aim_a[0, 0].item(), -1, 1)) * MAX_TURN_DEG
        dpitch = float(np.clip(aim_a[0, 1].item(), -1, 1)) * MAX_TURN_DEG

        send_frame(f, TYPE_ACTION, struct.pack(">ffBB", dyaw, dpitch, attack, forward))

        infer_ms += (time.perf_counter() - t0) * 1000
        n_ticks += 1
        n_attacks += attack
        n_forward += forward
        if n_ticks % 200 == 0:
            print(f"  {n_ticks} ticks | attack {n_attacks / 200:.0%} "
                  f"forward {n_forward / 200:.0%} | {infer_ms / 200:.1f} ms/tick")
            n_attacks = n_forward = 0
            infer_ms = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aim", default=str(MODELS / "aim.pt"))
    ap.add_argument("--swing", default=str(MODELS / "swing.pt"))
    ap.add_argument("--move", default=str(MODELS / "move.pt"))
    ap.add_argument("--port", type=int, default=36566)
    args = ap.parse_args()

    torch.set_num_threads(2)
    aim, shape = load_policy(AimPolicy, args.aim, "aim")
    if aim is None:
        raise SystemExit("the aim checkpoint is required")
    swing, _ = load_policy(SwingPolicy, args.swing, "swing")
    move, _ = load_policy(MovePolicy, args.move, "move")

    srv = socket.create_server(("127.0.0.1", args.port))
    print(f"pilot server listening on 127.0.0.1:{args.port} — "
          f"press the pilot key (default G) in game")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            serve_connection(conn, aim, swing, move, shape)
        except (ConnectionError, OSError) as e:
            print(f"client disconnected: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
