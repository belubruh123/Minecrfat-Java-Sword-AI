"""Single-env (n=1) adapter that makes the live pilot socket look exactly like
MinecraftVecEnv to PPOTrainer, so the SAME PPO code fine-tunes the fighter from
real in-game play.

The mod's client pilot bridge connects OUT to trainer/pilot_train.py (reverse of
the training bridge). In train mode the mod sends, per client tick, one obs
frame carrying mask + scalars + reward + done + info (pilot protocol v2), and we
send back one 13-byte action — strict alternation, i.e. lock-step, just like the
training bridge. Reward/done/info are computed client-side in PilotController
(mirroring Arena.stepCombo), so PPO sees the same (obs, action, reward, done)
tuples it would headless — only n=1 and at real time.

Correctness note: hit rewards are attributed on the tick the server CONFIRMS the
hit (a target hurtTime jump), which on a remote server lags the swing by up to
~10 ticks. That reward-attribution delay is harmless to PPO (GAE/returns absorb
it); it just means credit for a hit can land a few ticks after the swing.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .vec_env import (INFO_CHAIN_HIT, INFO_CRIT, INFO_HIT_LANDED,
                      INFO_HIT_TAKEN, INFO_SPRINT_HIT, INFO_WHIFF, MAX_TURN_DEG,
                      MinecraftVecEnv)

TYPE_JSON, TYPE_ACTION, TYPE_OBS = 0, 1, 2


def _read_frame(f, expected_type):
    hdr = f.read(5)
    if len(hdr) < 5:
        raise ConnectionError("pilot client closed")
    length, ftype = struct.unpack(">IB", hdr)
    payload = f.read(length - 1)
    if len(payload) < length - 1:
        raise ConnectionError("pilot client closed mid-frame")
    if ftype != expected_type:
        raise ConnectionError(f"expected frame type {expected_type}, got {ftype}")
    return payload


def _send_frame(f, ftype, payload):
    f.write(struct.pack(">IB", len(payload) + 1, ftype))
    f.write(payload)
    f.flush()


class PilotEnv:
    """MinecraftVecEnv-compatible view of one live player (n = 1).

    Implements the exact surface PPOTrainer uses: observe(), step(...),
    drain_stats(), and the attributes n / stack / h / w / n_scalars /
    curriculum. Downsampling, frame stacking and scalar normalization match
    MinecraftVecEnv so a fighter2 checkpoint sees identical inputs.
    """

    def __init__(self, conn, downsample: int = 3, frame_stack: int = 4,
                 curriculum: dict | None = None):
        self.n = 1
        self.stack = frame_stack
        self.ds = downsample
        self.n_scalars = 9
        self.curriculum = dict(curriculum or {"allow_jump": 0})

        self.conn = conn  # TCP_NODELAY set by the caller (pilot_train.py)
        self.f = conn.makefile("rwb")

        hello = json.loads(_read_frame(self.f, TYPE_JSON))
        self.width, self.height = hello["width"], hello["height"]
        self.h = self.height // downsample
        self.w = self.width // downsample
        if self.width // self.ds != self.w or self.height // self.ds != self.h:
            raise ConnectionError(
                f"obs {self.width}x{self.height} does not downsample to "
                f"{self.w}x{self.h} (ds={self.ds})")
        # ask for training mode: the mod then sends reward + done + info
        _send_frame(self.f, TYPE_JSON, b'{"msg":"ready","train":true}')
        self._mask_bytes = (self.width * self.height + 7) // 8

        # per-episode bookkeeping (mirrors MinecraftVecEnv)
        self.frames = np.zeros((1, frame_stack, self.h, self.w), np.uint8)
        self.episode_return = np.zeros(1, np.float64)
        self.episode_len = np.zeros(1, np.int64)
        self.episode_hits = np.zeros(1, np.int64)
        self.episode_whiffs = np.zeros(1, np.int64)
        self.episode_hits_taken = np.zeros(1, np.int64)
        self.episode_crits = np.zeros(1, np.int64)
        self.episode_sprint_hits = np.zeros(1, np.int64)
        self.episode_chain_hits = np.zeros(1, np.int64)
        self.finished: list[dict] = []

        mask, scalars, _, _, _ = self._recv_obs()  # initial obs (reward ignored)
        self.frames[:] = mask
        self.scalars = MinecraftVecEnv._norm_scalars(scalars)[None]  # (1, 9)

    # -- obs parsing --------------------------------------------------------

    def _recv_obs(self):
        payload = _read_frame(self.f, TYPE_OBS)
        mb = self._mask_bytes
        bits = np.unpackbits(np.frombuffer(payload, np.uint8, mb))
        full = bits[: self.height * self.width].reshape(self.height, self.width)
        small = full.reshape(self.h, self.ds, self.w, self.ds).max(axis=(1, 3))
        raw = np.frombuffer(payload, ">f4", self.n_scalars, mb).astype(np.float32)
        off = mb + 4 * self.n_scalars
        (reward,) = struct.unpack_from(">f", payload, off)
        done = payload[off + 4] != 0
        info = payload[off + 5]
        return small.astype(np.uint8), raw, np.float32(reward), bool(done), int(info)

    # -- MinecraftVecEnv surface -------------------------------------------

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return self.frames, self.scalars

    def step(self, dyaw, dpitch, attack=None, forward=None, jump=None,
             sprint=None, move=None, strafe=None):
        if move is None:
            move = np.zeros(1, np.uint8) if forward is None else np.asarray(forward)
        move = np.asarray(move)
        strafe = np.zeros(1, np.uint8) if strafe is None else np.asarray(strafe)
        if sprint is None:
            sprint = (move == 1).astype(np.uint8)
        attack = np.zeros(1, np.uint8) if attack is None else np.asarray(attack)
        jump = np.zeros(1, np.uint8) if jump is None else np.asarray(jump)

        def _f(x):  # first scalar of a length-1 array/scalar
            return float(np.asarray(x).reshape(-1)[0])

        def _i(x):
            return int(np.asarray(x).reshape(-1)[0])

        dyaw_deg = float(np.clip(_f(dyaw), -1, 1) * MAX_TURN_DEG)
        dpitch_deg = float(np.clip(_f(dpitch), -1, 1) * MAX_TURN_DEG)
        # pilot action order (mirrors trainer/pilot.py): dyaw, dpitch, attack,
        # move, strafe, jump, sprint
        _send_frame(self.f, TYPE_ACTION, struct.pack(
            ">ffBBBBB", dyaw_deg, dpitch_deg, _i(attack), _i(move),
            _i(strafe), _i(jump), _i(sprint)))

        mask, raw, reward, done, info = self._recv_obs()

        self.episode_return[0] += reward
        self.episode_len[0] += 1
        self.episode_hits[0] += (info & INFO_HIT_LANDED) > 0
        self.episode_whiffs[0] += (info & INFO_WHIFF) > 0
        self.episode_hits_taken[0] += (info & INFO_HIT_TAKEN) > 0
        self.episode_crits[0] += (info & INFO_CRIT) > 0
        self.episode_sprint_hits[0] += (info & INFO_SPRINT_HIT) > 0
        self.episode_chain_hits[0] += (info & INFO_CHAIN_HIT) > 0
        if done:
            self.finished.append({
                "arena": 0,
                "return": float(self.episode_return[0]),
                "length": int(self.episode_len[0]),
                "success": bool(reward > 1.0),  # kill bonus present
                "elevated": False,
                "hits": int(self.episode_hits[0]),
                "whiffs": int(self.episode_whiffs[0]),
                "hits_taken": int(self.episode_hits_taken[0]),
                "crits": int(self.episode_crits[0]),
                "sprint_hits": int(self.episode_sprint_hits[0]),
                "chain_hits": int(self.episode_chain_hits[0]),
            })
            self.episode_return[0] = 0.0
            self.episode_len[0] = 0
            self.episode_hits[0] = self.episode_whiffs[0] = 0
            self.episode_hits_taken[0] = self.episode_crits[0] = 0
            self.episode_sprint_hits[0] = self.episode_chain_hits[0] = 0

        self.frames = np.roll(self.frames, -1, axis=1)
        self.frames[0, -1] = mask
        if done:
            self.frames[0, :] = mask
        self.scalars = MinecraftVecEnv._norm_scalars(raw)[None]  # (1, 9)

        return (np.array([reward], np.float32),
                np.array([done], bool),
                np.array([info], np.uint8))

    def drain_stats(self) -> list[dict]:
        out = self.finished
        self.finished = []
        return out

    def close(self) -> None:
        try:
            self.f.close()
        finally:
            self.conn.close()
