"""Vectorized environment over the mod bridge: frame stacking, downsampling,
scalar normalization, per-arena episode bookkeeping, arena-0 recording."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .protocol import BridgeConnection

MAX_TURN_DEG = 15.0

INFO_ON_TARGET = 1
INFO_HIT_LANDED = 2
INFO_HIT_TAKEN = 4
INFO_ELEVATED = 8
INFO_WHIFF = 16
INFO_CRIT = 32
INFO_SPRINT_HIT = 64
INFO_CHAIN_HIT = 128  # landed hit extended a combo (chain >= 2)

SCALAR_SCALE = np.array([1 / 0.3, 1, 1, 1, 1 / 3.0, 1, 1, 1, 1 / 3.5],
                        dtype=np.float32)


class MinecraftVecEnv:
    """Steps all arenas at once. Observations: (n, stack, H, W) uint8 masks
    (downsampled with max-pooling so 1px silhouettes survive) + normalized
    scalars (n, 9)."""

    def __init__(self, curriculum: dict, stage: str = "aim", seed: int = 0,
                 host: str = "127.0.0.1", port: int = 36565,
                 downsample: int = 2, frame_stack: int = 4,
                 record_dir: str | None = None, record_arena: int = 0,
                 max_recorded_episodes: int = 30):
        self.conn = BridgeConnection(host=host, port=port)
        self.conn.configure(stage=stage, curriculum=curriculum, seed=seed,
                            record_arena=record_arena)
        self.n = self.conn.hello.arenas
        self.ds = downsample
        self.h = self.conn.hello.height // downsample
        self.w = self.conn.hello.width // downsample
        self.stack = frame_stack
        self.n_scalars = len(self.conn.hello.scalars)

        self.frames = np.zeros((self.n, frame_stack, self.h, self.w), np.uint8)
        self.episode_return = np.zeros(self.n, np.float64)
        self.episode_len = np.zeros(self.n, np.int64)
        self.episode_hits = np.zeros(self.n, np.int64)
        self.episode_whiffs = np.zeros(self.n, np.int64)
        self.episode_hits_taken = np.zeros(self.n, np.int64)
        self.episode_crits = np.zeros(self.n, np.int64)
        self.episode_sprint_hits = np.zeros(self.n, np.int64)
        self.episode_chain_hits = np.zeros(self.n, np.int64)
        self.finished: list[dict] = []  # stats of episodes done since last drain

        self.record_arena = record_arena
        self.record_dir = Path(record_dir) if record_dir else None
        self.max_recorded = max_recorded_episodes
        self._rec: dict[str, list] = {"masks": [], "actions": [], "rewards": [],
                                      "scalars": [], "infos": [], "telemetry": []}
        self._rec_counter = 0
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)

        obs = self.conn.recv_obs()  # initial observation
        masks = self._downsample(obs.masks)
        for i in range(frame_stack):
            self.frames[:, i] = masks
        self.scalars = self._norm_scalars(obs.scalars)

    def _downsample(self, masks: np.ndarray) -> np.ndarray:
        if self.ds == 1:
            return masks
        n, h, w = masks.shape
        return masks.reshape(n, self.h, self.ds, self.w, self.ds).max(axis=(2, 4))

    @staticmethod
    def _norm_scalars(raw: np.ndarray) -> np.ndarray:
        return np.clip(raw * SCALAR_SCALE, -3, 3).astype(np.float32)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return self.frames, self.scalars

    def step(self, dyaw: np.ndarray, dpitch: np.ndarray,
             attack: np.ndarray | None = None, forward: np.ndarray | None = None,
             jump: np.ndarray | None = None, sprint: np.ndarray | None = None,
             move: np.ndarray | None = None, strafe: np.ndarray | None = None):
        """Actions in [-1,1] for the rotation axes; returns (rewards, dones, infos).
        Pre-fighter stages pass `forward` (bool W) and it maps to move=1;
        sprint defaults to forward — those stages trained with W implying
        sprint, so their semantics are preserved when sprint isn't passed.
        The fighter stage passes move (0/1/2), strafe (0/1/2), jump, sprint
        explicitly."""
        zeros = np.zeros(self.n, np.uint8)
        if move is None:
            move = zeros if forward is None else forward
        strafe = zeros if strafe is None else strafe
        if sprint is None:
            sprint = (np.asarray(move) == 1).astype(np.uint8)
        attack = zeros if attack is None else attack
        jump = zeros if jump is None else jump
        self.conn.send_actions(
            np.clip(dyaw, -1, 1) * MAX_TURN_DEG,
            np.clip(dpitch, -1, 1) * MAX_TURN_DEG,
            attack, move, jump, sprint, strafe=strafe)
        obs = self.conn.recv_obs()

        masks = self._downsample(obs.masks)
        dones = obs.dones

        if self.record_dir is not None:
            a = self.record_arena
            self._rec["masks"].append(np.packbits(masks[a]))
            self._rec["actions"].append([float(dyaw[a]), float(dpitch[a]),
                                         float(attack[a]), float(move[a]),
                                         float(strafe[a]), float(jump[a]),
                                         float(sprint[a])])
            self._rec["rewards"].append(float(obs.rewards[a]))
            self._rec["scalars"].append(obs.scalars[a].copy())
            self._rec["infos"].append(int(obs.infos[a]))
            self._rec["telemetry"].append(obs.telemetry[a].copy())
            if dones[a]:
                self._flush_recording()

        self.episode_return += obs.rewards
        self.episode_len += 1
        self.episode_hits += (obs.infos & INFO_HIT_LANDED) > 0
        self.episode_whiffs += (obs.infos & INFO_WHIFF) > 0
        self.episode_hits_taken += (obs.infos & INFO_HIT_TAKEN) > 0
        self.episode_crits += (obs.infos & INFO_CRIT) > 0
        self.episode_sprint_hits += (obs.infos & INFO_SPRINT_HIT) > 0
        self.episode_chain_hits += (obs.infos & INFO_CHAIN_HIT) > 0
        for i in np.nonzero(dones)[0]:
            self.finished.append({
                "arena": int(i),
                "return": float(self.episode_return[i]),
                "length": int(self.episode_len[i]),
                "success": bool(obs.rewards[i] > 1.0),  # lock bonus present
                "elevated": bool(obs.infos[i] & INFO_ELEVATED),
                "hits": int(self.episode_hits[i]),
                "whiffs": int(self.episode_whiffs[i]),
                "hits_taken": int(self.episode_hits_taken[i]),
                "crits": int(self.episode_crits[i]),
                "sprint_hits": int(self.episode_sprint_hits[i]),
                "chain_hits": int(self.episode_chain_hits[i]),
            })
            self.episode_return[i] = 0.0
            self.episode_len[i] = 0
            self.episode_hits[i] = 0
            self.episode_whiffs[i] = 0
            self.episode_hits_taken[i] = 0
            self.episode_crits[i] = 0
            self.episode_sprint_hits[i] = 0
            self.episode_chain_hits[i] = 0

        # done obs is the first frame of the new episode: restart its stack
        self.frames = np.roll(self.frames, -1, axis=1)
        self.frames[:, -1] = masks
        for i in np.nonzero(dones)[0]:
            self.frames[i, :] = masks[i]
        self.scalars = self._norm_scalars(obs.scalars)

        return obs.rewards, dones, obs.infos

    def drain_stats(self) -> list[dict]:
        out = self.finished
        self.finished = []
        return out

    def reconfigure(self, stage: str, curriculum: dict, seed: int = 0) -> None:
        self.conn.send_json({"msg": "config", "stage": stage, "seed": seed,
                             "curriculum": curriculum,
                             "record_arena": self.record_arena})

    def _flush_recording(self) -> None:
        n_frames = len(self._rec["rewards"])
        if n_frames < 2:
            self._reset_recording()
            return
        name = f"ep_{int(time.time() * 1000) % 10**10}_{self._rec_counter:04d}"
        self._rec_counter += 1
        np.savez_compressed(
            self.record_dir / f"{name}.npz",
            masks=np.stack(self._rec["masks"]),
            actions=np.array(self._rec["actions"], np.float32),
            rewards=np.array(self._rec["rewards"], np.float32),
            scalars=np.stack(self._rec["scalars"]),
            infos=np.array(self._rec["infos"], np.uint8),
            telemetry=np.stack(self._rec["telemetry"]),
            shape=np.array([self.h, self.w]),
        )
        meta = {"name": name, "frames": n_frames,
                "return": float(sum(self._rec["rewards"])), "ts": time.time()}
        (self.record_dir / f"{name}.json").write_text(json.dumps(meta))
        self._reset_recording()
        self._prune_recordings()

    def _reset_recording(self) -> None:
        for v in self._rec.values():
            v.clear()

    def _prune_recordings(self) -> None:
        eps = sorted(self.record_dir.glob("ep_*.npz"))
        for old in eps[: max(0, len(eps) - self.max_recorded)]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    def close(self) -> None:
        self.conn.close()
