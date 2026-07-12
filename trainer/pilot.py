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
import random
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch

from drlagent.models import (AimPolicy, ComboPolicy, Fighter2Policy,
                             FighterPolicy, MovePolicy, SwingPolicy)
from drlagent.vec_env import MAX_TURN_DEG, MinecraftVecEnv

TYPE_JSON, TYPE_ACTION, TYPE_OBS = 0, 1, 2
MODELS = Path(__file__).resolve().parent.parent / "models"


class AimSmoother:
    """The aim net is saturated bang-bang control: it slams the ±15°/tick
    clamp and flips direction ~72% of ticks — dead-on target on average, but
    on screen it shakes unnaturally. A two-tap average cancels perfect
    tick-to-tick alternation exactly (the tracking component passes with one
    tick of lag), a light low-pass eats the residue, and a rate cap keeps
    flicks at human mouse speeds."""

    MAX_RATE = 11.0  # deg/tick output cap (vanilla clamp is 15)

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.prev_cmd = np.zeros(2)
        self.state = np.zeros(2)

    def __call__(self, dyaw: float, dpitch: float) -> tuple[float, float]:
        cmd = np.array([dyaw, dpitch])
        two_tap = 0.5 * (cmd + self.prev_cmd)
        self.prev_cmd = cmd
        self.state += self.alpha * (two_tap - self.state)
        out = np.clip(self.state, -self.MAX_RATE, self.MAX_RATE)
        return float(out[0]), float(out[1])


class InputHumanizer:
    """Discrete inputs at human timescales. The policy re-decides every key
    20x a second; a person holds keys for hundreds of ms, doesn't re-toggle
    sprint tick by tick, clicks with a little timing jitter, and can't click
    faster than ~7/s. A key change is adopted only after the current state
    has been held for its minimum time; attacks get 0-2 ticks of jitter and
    a minimum re-click gap."""

    MIN_HOLD = {"move": 4, "strafe": 3, "jump": 2, "sprint": 6}  # ticks
    MIN_CLICK_GAP = 3

    def __init__(self):
        self.rng = random.Random()
        self.state = {"move": 0, "strafe": 0, "jump": 0, "sprint": 0}
        self.held = {k: 99 for k in self.state}
        self.click_delay = -1  # -1 = no click pending
        self.since_click = 99

    def keys(self, move: int, strafe: int, jump: int, sprint: int):
        want = {"move": move, "strafe": strafe, "jump": jump, "sprint": sprint}
        for k, v in want.items():
            self.held[k] += 1
            if v != self.state[k] and self.held[k] >= self.MIN_HOLD[k]:
                self.state[k] = v
                self.held[k] = 0
        s = self.state
        return s["move"], s["strafe"], s["jump"], s["sprint"]

    def attack(self, want: int) -> int:
        self.since_click += 1
        if want and self.click_delay < 0 and self.since_click >= self.MIN_CLICK_GAP:
            self.click_delay = self.rng.choice((0, 1, 1, 2))
        if self.click_delay < 0:
            return 0
        if self.click_delay == 0:
            self.click_delay = -1
            self.since_click = 0
            return 1
        self.click_delay -= 1
        return 0


def load_policy(cls, path: str | None, name: str):
    if not path:
        print(f"{name}: disabled (no checkpoint)")
        return None, None
    p = Path(path)
    if not p.exists():
        print(f"{name}: disabled ({p} does not exist)")
        return None, None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if cls is None:  # movement head: the checkpoint's stage picks the class
        cls = {"combo": ComboPolicy, "fighter": FighterPolicy,
               "fighter2": Fighter2Policy}.get(ckpt.get("stage"), MovePolicy)
    net = cls(*ckpt["obs_shape"])
    net.load_state_dict(ckpt["policy"])
    net.eval()
    net.requires_grad_(False)
    # models trained with jump as a no-op have arbitrary jump-bit logits —
    # the pilot must suppress the key or it would leap around meaninglessly
    net.allow_jump = bool(ckpt.get("allow_jump", True))
    print(f"{name}: {p} ({ckpt['step']} steps, stage {ckpt.get('stage', '?')}"
          f"{'' if net.allow_jump else ', no-jump'})")
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


def serve_connection(conn, aim, swing, move, shape, smooth=True, humanize=True):
    stack, h, w, n_scalars = shape
    smoother = AimSmoother() if smooth else None
    humanizer = InputHumanizer() if humanize else None
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
            attack = mv = strafe = jump = sprint = 0
            if swing is not None:
                attack = int(swing.act(tm, ts)[0].item())
            if move is not None:
                a = int(move.act(tm, ts)[0].item())
                if isinstance(move, Fighter2Policy):
                    mv, strafe, jump, sprint, attack = (
                        int(x) for x in Fighter2Policy.decode(a))
                    if not getattr(move, "allow_jump", True):
                        jump = 0
                elif isinstance(move, FighterPolicy):
                    mv, strafe, jump, sprint = (int(x) for x in FighterPolicy.decode(a))
                    if not getattr(move, "allow_jump", True):
                        jump = 0
                elif isinstance(move, ComboPolicy):
                    mv, jump, sprint = a & 1, (a >> 1) & 1, (a >> 2) & 1
                else:  # move stage trained with W implying sprint
                    mv, sprint = a, a
        dyaw = float(np.clip(aim_a[0, 0].item(), -1, 1)) * MAX_TURN_DEG
        dpitch = float(np.clip(aim_a[0, 1].item(), -1, 1)) * MAX_TURN_DEG
        if smoother is not None:
            dyaw, dpitch = smoother(dyaw, dpitch)
        if humanizer is not None:
            mv, strafe, jump, sprint = humanizer.keys(mv, strafe, jump, sprint)
            attack = humanizer.attack(attack)

        send_frame(f, TYPE_ACTION, struct.pack(">ffBBBBB", dyaw, dpitch, attack,
                                               mv, strafe, jump, sprint))

        infer_ms += (time.perf_counter() - t0) * 1000
        n_ticks += 1
        n_attacks += attack
        n_forward += int(mv == 1)
        if n_ticks % 200 == 0:
            print(f"  {n_ticks} ticks | attack {n_attacks / 200:.0%} "
                  f"forward {n_forward / 200:.0%} | {infer_ms / 200:.1f} ms/tick")
            n_attacks = n_forward = 0
            infer_ms = 0.0


def main():
    ap = argparse.ArgumentParser()
    default_move = next((MODELS / n for n in ("fighter2.pt", "fighter.pt", "combo.pt", "move.pt")
                         if (MODELS / n).exists()), MODELS / "move.pt")
    ap.add_argument("--aim", default=str(MODELS / "aim.pt"))
    ap.add_argument("--swing", default=str(MODELS / "swing.pt"))
    ap.add_argument("--move", default=str(default_move),
                    help="movement checkpoint: a move- or combo-stage .pt")
    ap.add_argument("--port", type=int, default=36566)
    ap.add_argument("--raw-aim", action="store_true",
                    help="disable aim smoothing (the raw policy shakes)")
    ap.add_argument("--raw-keys", action="store_true",
                    help="disable input humanizing (per-tick key retoggling)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    aim, shape = load_policy(AimPolicy, args.aim, "aim")
    if aim is None:
        raise SystemExit("the aim checkpoint is required")
    swing, _ = load_policy(SwingPolicy, args.swing, "swing")
    move, _ = load_policy(None, args.move, "move")

    srv = socket.create_server(("127.0.0.1", args.port))
    print(f"pilot server listening on 127.0.0.1:{args.port} — "
          f"press the pilot key (default G) in game")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            serve_connection(conn, aim, swing, move, shape,
                             smooth=not args.raw_aim,
                             humanize=not args.raw_keys)
        except (ConnectionError, OSError) as e:
            print(f"client disconnected: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
