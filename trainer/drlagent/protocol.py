"""Wire protocol for the mod bridge. See PROTOCOL.md at the repo root."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass

import numpy as np

TYPE_JSON = 0
TYPE_ACTIONS = 1
TYPE_OBS = 2

DEFAULT_PORT = 36565


@dataclass
class Hello:
    protocol: int
    mc_version: str
    arenas: int
    width: int
    height: int
    scalars: list[str]


N_TELEMETRY = 12  # ax, az, ay, ayaw, ox, oz, oy, apitch, oyaw,
                 # agent_swung, opp_swung, opp_hurt (replay/debug only)


@dataclass
class ObsBatch:
    tick: int
    masks: np.ndarray    # (arenas, H, W) uint8, values 0/1
    scalars: np.ndarray  # (arenas, n_scalars) f32
    rewards: np.ndarray  # (arenas,) f32
    dones: np.ndarray    # (arenas,) bool
    infos: np.ndarray    # (arenas,) uint8 bit flags
    telemetry: np.ndarray  # (arenas, N_TELEMETRY) f32 — never fed to policies


class BridgeConnection:
    """Blocking lock-step connection to the mod."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 connect_timeout: float = 60.0):
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.settimeout(None)  # lock-step steps may legitimately take long
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.hello = self._recv_hello()
        self._mask_bytes = (self.hello.width * self.hello.height + 7) // 8
        self._n_scalars = len(self.hello.scalars)

    # -- framing ------------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            r = self.sock.recv_into(view[got:], n - got)
            if r == 0:
                raise ConnectionError("bridge closed by mod")
            got += r
        return bytes(buf)

    def _recv_frame(self) -> tuple[int, bytes]:
        (length,) = struct.unpack(">I", self._recv_exact(4))
        body = self._recv_exact(length)
        return body[0], body[1:]

    def _send_frame(self, msg_type: int, payload: bytes) -> None:
        self.sock.sendall(struct.pack(">IB", len(payload) + 1, msg_type) + payload)

    # -- control channel ----------------------------------------------------

    def send_json(self, obj: dict) -> None:
        self._send_frame(TYPE_JSON, json.dumps(obj).encode())

    def recv_json(self) -> dict:
        msg_type, payload = self._recv_frame()
        if msg_type != TYPE_JSON:
            raise ProtocolError(f"expected JSON frame, got type {msg_type}")
        return json.loads(payload)

    def _recv_hello(self) -> Hello:
        msg = self.recv_json()
        if msg.get("msg") != "hello":
            raise ProtocolError(f"expected hello, got {msg}")
        return Hello(
            protocol=msg["protocol"],
            mc_version=msg["mc_version"],
            arenas=msg["arenas"],
            width=msg["width"],
            height=msg["height"],
            scalars=msg["scalars"],
        )

    def configure(self, stage: str, curriculum: dict, seed: int = 0,
                  record_arena: int = 0) -> None:
        self.send_json({
            "msg": "config",
            "stage": stage,
            "seed": seed,
            "curriculum": curriculum,
            "record_arena": record_arena,
        })
        reply = self.recv_json()
        if reply.get("msg") != "ready":
            raise ProtocolError(f"expected ready, got {reply}")

    # -- hot path -----------------------------------------------------------

    def send_actions(self, dyaw: np.ndarray, dpitch: np.ndarray,
                     attack: np.ndarray, move: np.ndarray,
                     jump: np.ndarray, sprint: np.ndarray,
                     reset: np.ndarray | None = None,
                     strafe: np.ndarray | None = None) -> None:
        """v3 actions. move: 0 none / 1 W / 2 S (old boolean forward still
        works: True == 1 == W). strafe: 0 none / 1 A / 2 D."""
        n = self.hello.arenas
        if reset is None:
            reset = np.zeros(n, dtype=np.uint8)
        if strafe is None:
            strafe = np.zeros(n, dtype=np.uint8)
        payload = bytearray()
        for i in range(n):
            payload += struct.pack(">ffBBBBBB", float(dyaw[i]), float(dpitch[i]),
                                   int(attack[i]), int(move[i]), int(strafe[i]),
                                   int(jump[i]), int(sprint[i]), int(reset[i]))
        self._send_frame(TYPE_ACTIONS, bytes(payload))

    def recv_obs(self) -> ObsBatch:
        msg_type, payload = self._recv_frame()
        if msg_type != TYPE_OBS:
            raise ProtocolError(f"expected obs frame, got type {msg_type}")
        h, w, n = self.hello.height, self.hello.width, self.hello.arenas
        per_arena = (self._mask_bytes + 4 * self._n_scalars + 4 + 1 + 1
                     + 4 * N_TELEMETRY)
        (tick,) = struct.unpack(">I", payload[:4])
        masks = np.empty((n, h, w), dtype=np.uint8)
        scalars = np.empty((n, self._n_scalars), dtype=np.float32)
        rewards = np.empty(n, dtype=np.float32)
        dones = np.empty(n, dtype=bool)
        infos = np.empty(n, dtype=np.uint8)
        telemetry = np.empty((n, N_TELEMETRY), dtype=np.float32)
        off = 4
        for i in range(n):
            end = off + self._mask_bytes
            bits = np.unpackbits(np.frombuffer(payload, np.uint8, self._mask_bytes, off))
            masks[i] = bits[: h * w].reshape(h, w)
            off = end
            scalars[i] = np.frombuffer(payload, ">f4", self._n_scalars, off)
            off += 4 * self._n_scalars
            (rewards[i],) = struct.unpack_from(">f", payload, off)
            off += 4
            dones[i] = payload[off] != 0
            infos[i] = payload[off + 1]
            off += 2
            telemetry[i] = np.frombuffer(payload, ">f4", N_TELEMETRY, off)
            off += 4 * N_TELEMETRY
        if off != len(payload) or per_arena * n + 4 != len(payload):
            raise ProtocolError(f"obs frame size mismatch: read {off}, got {len(payload)}")
        return ObsBatch(tick, masks, scalars, rewards, dones, infos, telemetry)

    def close(self) -> None:
        self.sock.close()


class ProtocolError(Exception):
    pass
