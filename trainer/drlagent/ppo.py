"""CleanRL-style PPO for the aim policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .models import AimPolicy
from .vec_env import MinecraftVecEnv


class PPOTrainer:
    def __init__(self, env: MinecraftVecEnv, run_dir: Path, cfg: dict):
        self.env = env
        self.cfg = cfg
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        torch.set_num_threads(cfg.get("torch_threads", 3))

        self.policy = AimPolicy(env.stack, env.h, env.w, env.n_scalars)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=cfg["lr"], eps=1e-5)
        self.writer = SummaryWriter(str(run_dir / "tb"))
        self.metrics_file = (run_dir / "metrics.jsonl").open("a")

        self.rollout_len = cfg["rollout_len"]
        n = env.n
        t = self.rollout_len
        self.b_masks = np.zeros((t, n, env.stack, env.h, env.w), np.uint8)
        self.b_scalars = np.zeros((t, n, env.n_scalars), np.float32)
        self.b_actions = np.zeros((t, n, 2), np.float32)
        self.b_logprobs = np.zeros((t, n), np.float32)
        self.b_rewards = np.zeros((t, n), np.float32)
        self.b_dones = np.zeros((t, n), np.float32)
        self.b_values = np.zeros((t, n), np.float32)

        self.global_step = 0
        self.update_idx = 0
        self.ep_stats: list[dict] = []

    # -- data collection ------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(self) -> None:
        env = self.env
        reward_scale = self.cfg.get("reward_scale", 1.0)
        for t in range(self.rollout_len):
            masks, scalars = env.observe()
            self.b_masks[t] = masks
            self.b_scalars[t] = scalars
            tm = torch.from_numpy(masks).float()
            ts = torch.from_numpy(scalars)
            action, logprob, _, value = self.policy.act(tm, ts)
            a = action.numpy()
            rewards, dones, _ = env.step(a[:, 0], a[:, 1])
            self.b_actions[t] = a
            self.b_logprobs[t] = logprob.numpy()
            self.b_values[t] = value.numpy()
            self.b_rewards[t] = rewards * reward_scale
            self.b_dones[t] = dones
            self.global_step += env.n
        self.ep_stats.extend(env.drain_stats())

    @torch.no_grad()
    def _bootstrap(self) -> tuple[np.ndarray, np.ndarray]:
        masks, scalars = self.env.observe()
        next_value = self.policy.value(torch.from_numpy(masks).float(),
                                       torch.from_numpy(scalars)).numpy()
        gamma, lam = self.cfg["gamma"], self.cfg["gae_lambda"]
        t_len, n = self.rollout_len, self.env.n
        advantages = np.zeros((t_len, n), np.float32)
        lastgae = np.zeros(n, np.float32)
        for t in reversed(range(t_len)):
            if t == t_len - 1:
                nextnonterminal = 1.0 - self.b_dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.b_dones[t]
                nextvalues = self.b_values[t + 1]
            delta = (self.b_rewards[t] + gamma * nextvalues * nextnonterminal
                     - self.b_values[t])
            lastgae = delta + gamma * lam * nextnonterminal * lastgae
            advantages[t] = lastgae
        returns = advantages + self.b_values
        return advantages, returns

    # -- optimization ---------------------------------------------------------

    def update(self) -> dict:
        cfg = self.cfg
        advantages, returns = self._bootstrap()
        t_len, n = self.rollout_len, self.env.n
        batch = t_len * n

        flat_masks = self.b_masks.reshape(batch, *self.b_masks.shape[2:])
        flat_scalars = self.b_scalars.reshape(batch, -1)
        flat_actions = torch.from_numpy(self.b_actions.reshape(batch, 2))
        flat_logprobs = torch.from_numpy(self.b_logprobs.reshape(batch))
        flat_adv = torch.from_numpy(advantages.reshape(batch))
        flat_returns = torch.from_numpy(returns.reshape(batch))
        flat_values = torch.from_numpy(self.b_values.reshape(batch))

        idx = np.arange(batch)
        mb_size = batch // cfg["num_minibatches"]
        clip = cfg["clip_coef"]
        pg_losses, v_losses, entropies, clipfracs = [], [], [], []

        for _ in range(cfg["update_epochs"]):
            np.random.shuffle(idx)
            for start in range(0, batch, mb_size):
                mb = idx[start:start + mb_size]
                tm = torch.from_numpy(flat_masks[mb]).float()
                ts = torch.from_numpy(flat_scalars[mb])
                _, newlogprob, entropy, newvalue = self.policy.act(
                    tm, ts, action=flat_actions[mb])
                logratio = newlogprob - flat_logprobs[mb]
                ratio = logratio.exp()
                clipfracs.append(((ratio - 1).abs() > clip).float().mean().item())

                mb_adv = flat_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clipped = flat_values[mb] + torch.clamp(
                    newvalue - flat_values[mb], -clip, clip)
                v_loss = 0.5 * torch.max(
                    (newvalue - flat_returns[mb]) ** 2,
                    (v_clipped - flat_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (pg_loss - cfg["ent_coef"] * entropy_loss
                        + cfg["vf_coef"] * v_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         cfg["max_grad_norm"])
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

        self.update_idx += 1
        return {"pg_loss": float(np.mean(pg_losses)),
                "v_loss": float(np.mean(v_losses)),
                "entropy": float(np.mean(entropies)),
                "clipfrac": float(np.mean(clipfracs))}

    # -- logging / checkpoints ------------------------------------------------

    def log(self, losses: dict, sps: float) -> dict:
        stats = self.ep_stats
        self.ep_stats = []
        summary = {
            "step": self.global_step,
            "update": self.update_idx,
            "sps": round(sps, 1),
            **losses,
        }
        if stats:
            summary["ep_return"] = float(np.mean([s["return"] for s in stats]))
            summary["ep_len"] = float(np.mean([s["length"] for s in stats]))
            summary["success"] = float(np.mean([s["success"] for s in stats]))
            summary["episodes"] = len(stats)
        for k, v in summary.items():
            if k not in ("step", "episodes"):
                self.writer.add_scalar(k, v, self.global_step)
        self.metrics_file.write(json.dumps(summary) + "\n")
        self.metrics_file.flush()
        return summary

    def save(self, name: str = "latest") -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "cfg": self.cfg,
            "obs_shape": [self.env.stack, self.env.h, self.env.w,
                          self.env.n_scalars],
        }, self.run_dir / f"{name}.pt")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]

    # -- main loop ------------------------------------------------------------

    def train(self, total_steps: int) -> None:
        while self.global_step < total_steps:
            t0 = time.time()
            self.collect_rollout()
            losses = self.update()
            sps = self.rollout_len * self.env.n / (time.time() - t0)
            summary = self.log(losses, sps)
            if self.update_idx % 10 == 0:
                print(f"[{self.global_step:>9}] " + " ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in summary.items() if k != "step"), flush=True)
            if self.update_idx % 50 == 0:
                self.save("latest")
        self.save("final")
