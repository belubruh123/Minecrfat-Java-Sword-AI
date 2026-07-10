"""Rehearsal curriculum with an anti-forgetting eval guard.

The aim policy is one network for both axes; stages only change the spawn
distribution. Later stages keep a rehearsal share of earlier-stage
(horizontal) episodes, and a periodic deterministic eval checks each
scenario class separately: if the earlier skill decays past the guard
threshold, its rehearsal share is increased.
"""

from __future__ import annotations

import numpy as np
import torch

from .vec_env import MinecraftVecEnv


class RehearsalCurriculum:
    def __init__(self, env: MinecraftVecEnv, base_curriculum: dict, stage: str,
                 eval_every: int = 50_000, eval_episodes: int = 48,
                 eval_seed: int = 777,
                 h_prob_floor: float = 0.25, h_prob_cap: float = 0.6,
                 guard_drop: float = 0.10):
        self.env = env
        self.base = dict(base_curriculum)
        self.stage = stage
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.h_prob = float(base_curriculum.get("horizontal_prob", 0.3))
        self.h_prob_floor = h_prob_floor
        self.h_prob_cap = h_prob_cap
        self.guard_drop = guard_drop
        self.best_h_success = 0.0
        self._next_eval = eval_every

    def maybe_eval(self, step: int, policy) -> dict | None:
        if step < self._next_eval:
            return None
        self._next_eval = step + self.eval_every
        stats = self._run_eval(policy)
        h = [s for s in stats if not s["elevated"]]
        v = [s for s in stats if s["elevated"]]
        h_succ = float(np.mean([s["success"] for s in h])) if h else float("nan")
        v_succ = float(np.mean([s["success"] for s in v])) if v else float("nan")

        adapted = False
        if h and h_succ > self.best_h_success:
            self.best_h_success = h_succ
        elif h and self.best_h_success - h_succ > self.guard_drop:
            new = min(self.h_prob + 0.1, self.h_prob_cap)
            if new != self.h_prob:
                self.h_prob = new
                adapted = True

        self._reconfigure_training()
        return {"eval_success_h": h_succ, "eval_success_v": v_succ,
                "h_prob": self.h_prob, "adapted": adapted,
                "eval_episodes_h": len(h), "eval_episodes_v": len(v)}

    @torch.no_grad()
    def _run_eval(self, policy) -> list[dict]:
        env = self.env
        # 50/50 split so both scenario classes get equal eval coverage,
        # reseeded for a reproducible spawn sequence.
        eval_cur = {**self.base, "horizontal_prob": 0.5}
        env.conn.send_json({"msg": "config", "stage": self.stage,
                            "seed": self.eval_seed, "reseed": True,
                            "curriculum": eval_cur})
        env.drain_stats()  # discard episodes cut short by the forced reset
        # the forced reset lands with the NEXT obs; step once to sync stacks
        n = env.n
        zeros = np.zeros(n, np.float32)
        env.step(zeros, zeros)
        env.drain_stats()

        stats: list[dict] = []
        while len(stats) < self.eval_episodes:
            masks, scalars = env.observe()
            action, _, _, _ = policy.act(torch.from_numpy(masks).float(),
                                         torch.from_numpy(scalars),
                                         deterministic=True)
            a = action.numpy()
            env.step(a[:, 0], a[:, 1])
            stats.extend(env.drain_stats())
        return stats

    def _reconfigure_training(self) -> None:
        cur = {**self.base, "horizontal_prob": self.h_prob}
        self.env.reconfigure(stage=self.stage, curriculum=cur)
