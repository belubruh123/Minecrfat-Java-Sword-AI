"""CleanRL-style PPO for the aim policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .models import (AimPolicy, ComboPolicy, Fighter2Policy, FighterPolicy,
                     MovePolicy, SwingPolicy)
from .vec_env import MinecraftVecEnv


def _load_frozen(cls, path: str, *shape):
    net = cls(*shape)
    net.load_state_dict(torch.load(path, map_location="cpu",
                                   weights_only=False)["policy"])
    net.eval()
    net.requires_grad_(False)
    return net


def _warm_start_trunk(policy: nn.Module, path: str) -> list[str]:
    """Copy every shape-matching parameter from a checkpoint (encoder,
    scalar_fc, critic — everything except a differently-shaped actor head).
    Returns the names that were skipped."""
    src = torch.load(path, map_location="cpu", weights_only=False)["policy"]
    dst = policy.state_dict()
    skipped = []
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
        else:
            skipped.append(k)
    policy.load_state_dict(dst)
    return skipped


class PPOTrainer:
    """Trains one policy per stage. "aim" trains the Gaussian aim net; "swing"
    trains a Categorical attack net while a frozen aim checkpoint steers;
    "move" trains the W key with frozen aim + swing acting."""

    def __init__(self, env: MinecraftVecEnv, run_dir: Path, cfg: dict,
                 stage: str = "aim", aim_ckpt: str | None = None,
                 swing_ckpt: str | None = None, init_ckpt: str | None = None):
        self.env = env
        self.cfg = cfg
        self.run_dir = run_dir
        self.stage = stage
        run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        torch.set_num_threads(cfg.get("torch_threads", 3))

        shape = (env.stack, env.h, env.w, env.n_scalars)
        self.aim = None
        self.swing = None
        if stage in ("swing", "move", "combo", "fighter", "fighter2"):
            if not aim_ckpt:
                raise ValueError(f"{stage} stage needs aim_checkpoint in the config")
            self.aim = _load_frozen(AimPolicy, aim_ckpt, *shape)
        if stage == "swing":
            self.policy = SwingPolicy(*shape)
            self.act_dim = 1
        elif stage == "fighter2":
            # fighter2 owns the attack button — no frozen swing net
            self.policy = Fighter2Policy(*shape)
            self.act_dim = 1
        elif stage in ("move", "combo", "fighter"):
            if not swing_ckpt:
                raise ValueError(f"{stage} stage needs swing_checkpoint in the config")
            self.swing = _load_frozen(SwingPolicy, swing_ckpt, *shape)
            self.policy = {"move": MovePolicy, "combo": ComboPolicy,
                           "fighter": FighterPolicy}[stage](*shape)
            self.act_dim = 1
        else:
            self.policy = AimPolicy(*shape)
            self.act_dim = 2
        if init_ckpt:
            skipped = _warm_start_trunk(self.policy, init_ckpt)
            print(f"warm-started trunk from {init_ckpt}"
                  f" (skipped: {', '.join(skipped) or 'nothing'})")
            # Under lock_pitch the dpitch action is forced to 0, so the pitch
            # output (actor_mean row 1 + its logstd) never sees a reward
            # gradient and drifts to garbage over the stage. Warm-starting the
            # vertical stage from it would inherit that garbage and break the
            # aim the instant pitch unlocks (even eye-level targets go to 0%).
            # Reset ONLY the pitch head: keep the trained yaw, start pitch at
            # ~0 with fresh exploration so the vertical reward can teach it.
            if cfg.get("reset_pitch_head") and isinstance(self.policy, AimPolicy):
                with torch.no_grad():
                    self.policy.actor_mean.weight[1].normal_(0.0, 0.01)
                    self.policy.actor_mean.bias[1].zero_()
                    self.policy.actor_logstd[0, 1] = -0.3
                print("reset pitch head (dpitch mean+logstd) for the vertical stage")
            # Symmetric: the pure-vertical stage HARD-locks yaw, so the yaw head
            # rots there. The both-axes stage resets it and relearns yaw on the
            # (now target-aware) encoder, keeping the good pitch head.
            if cfg.get("reset_yaw_head") and isinstance(self.policy, AimPolicy):
                with torch.no_grad():
                    self.policy.actor_mean.weight[0].normal_(0.0, 0.01)
                    self.policy.actor_mean.bias[0].zero_()
                    self.policy.actor_logstd[0, 0] = -0.3
                print("reset yaw head (dyaw mean+logstd) for the both-axes stage")
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=cfg["lr"], eps=1e-5)
        self.writer = SummaryWriter(str(run_dir / "tb"))
        self.metrics_file = (run_dir / "metrics.jsonl").open("a")

        # return-based best checkpoint: converged policies self-destruct if
        # trained past their plateau (advantages vanish, the entropy bonus
        # slowly randomizes the actor) — the peak artifact must survive that
        self._ret_hist: list[float] = []
        self._best_ret = float("-inf")

        self.rollout_len = cfg["rollout_len"]
        n = env.n
        t = self.rollout_len
        self.b_masks = np.zeros((t, n, env.stack, env.h, env.w), np.uint8)
        self.b_scalars = np.zeros((t, n, env.n_scalars), np.float32)
        self.b_actions = np.zeros((t, n, self.act_dim), np.float32)
        self.b_logprobs = np.zeros((t, n), np.float32)
        self.b_rewards = np.zeros((t, n), np.float32)
        self.b_dones = np.zeros((t, n), np.float32)
        self.b_values = np.zeros((t, n), np.float32)

        self.global_step = 0
        self.update_idx = 0
        self.best_eval = float("-inf")
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
            if self.stage == "swing":
                aim_a, _, _, _ = self.aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                a = action.numpy()
                rewards, dones, _ = env.step(aim_a[:, 0], aim_a[:, 1],
                                             attack=a.astype(np.uint8))
                self.b_actions[t] = a[:, None]
            elif self.stage == "move":
                aim_a, _, _, _ = self.aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                # the swing head is sampled: that is its trained behavior
                swing_a, _, _, _ = self.swing.act(tm, ts)
                a = action.numpy()
                rewards, dones, _ = env.step(
                    aim_a[:, 0], aim_a[:, 1],
                    attack=swing_a.numpy().astype(np.uint8),
                    forward=a.astype(np.uint8))
                self.b_actions[t] = a[:, None]
            elif self.stage == "combo":
                aim_a, _, _, _ = self.aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                swing_a, _, _, _ = self.swing.act(tm, ts)
                a = action.numpy().astype(np.int64)
                rewards, dones, _ = env.step(
                    aim_a[:, 0], aim_a[:, 1],
                    attack=swing_a.numpy().astype(np.uint8),
                    forward=(a & 1).astype(np.uint8),
                    jump=((a >> 1) & 1).astype(np.uint8),
                    sprint=((a >> 2) & 1).astype(np.uint8))
                self.b_actions[t] = a[:, None]
            elif self.stage == "fighter2":
                aim_a, _, _, _ = self.aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                a = action.numpy().astype(np.int64)
                move, strafe, jump, sprint, attack = Fighter2Policy.decode(a)
                rewards, dones, _ = env.step(
                    aim_a[:, 0], aim_a[:, 1],
                    attack=attack.astype(np.uint8),
                    move=move.astype(np.uint8),
                    strafe=strafe.astype(np.uint8),
                    jump=jump.astype(np.uint8),
                    sprint=sprint.astype(np.uint8))
                self.b_actions[t] = a[:, None]
            elif self.stage == "fighter":
                aim_a, _, _, _ = self.aim.act(tm, ts, deterministic=True)
                aim_a = aim_a.numpy()
                swing_a, _, _, _ = self.swing.act(tm, ts)
                a = action.numpy().astype(np.int64)
                move, strafe, jump, sprint = FighterPolicy.decode(a)
                rewards, dones, _ = env.step(
                    aim_a[:, 0], aim_a[:, 1],
                    attack=swing_a.numpy().astype(np.uint8),
                    move=move.astype(np.uint8),
                    strafe=strafe.astype(np.uint8),
                    jump=jump.astype(np.uint8),
                    sprint=sprint.astype(np.uint8))
                self.b_actions[t] = a[:, None]
            else:
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
        flat_actions = torch.from_numpy(self.b_actions.reshape(batch, self.act_dim))
        if self.stage in ("swing", "move", "combo", "fighter", "fighter2"):
            flat_actions = flat_actions.long().squeeze(-1)
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
            if self.stage in ("swing", "move", "combo", "fighter", "fighter2"):
                summary["ep_hits"] = float(np.mean([s["hits"] for s in stats]))
                summary["ep_whiffs"] = float(np.mean([s["whiffs"] for s in stats]))
            if self.stage in ("move", "combo", "fighter", "fighter2"):
                summary["ep_hits_taken"] = float(
                    np.mean([s["hits_taken"] for s in stats]))
            if self.stage in ("combo", "fighter", "fighter2"):
                summary["ep_crits"] = float(np.mean([s["crits"] for s in stats]))
                summary["ep_sprint_hits"] = float(
                    np.mean([s["sprint_hits"] for s in stats]))
                summary["ep_chain_hits"] = float(
                    np.mean([s.get("chain_hits", 0) for s in stats]))
        for k, v in summary.items():
            if k not in ("step", "episodes"):
                self.writer.add_scalar(k, v, self.global_step)
        r = summary.get("ep_return")
        if r is not None:
            self._ret_hist.append(float(r))
            if len(self._ret_hist) >= 3:
                m = float(np.mean(self._ret_hist[-5:]))
                if m > self._best_ret:
                    self._best_ret = m
                    self.save("best")
        self.metrics_file.write(json.dumps(summary) + "\n")
        self.metrics_file.flush()
        return summary

    def save(self, name: str = "latest") -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "cfg": self.cfg,
            "stage": self.stage,
            # no-jump curriculum: the pilot must suppress the jump bit for
            # models whose jump input was a training-time no-op
            "allow_jump": bool(getattr(self.env, "curriculum", {}).get("allow_jump", 1)),
            "obs_shape": [self.env.stack, self.env.h, self.env.w,
                          self.env.n_scalars],
        }, self.run_dir / f"{name}.pt")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]

    # -- main loop ------------------------------------------------------------

    def train(self, total_steps: int, curriculum=None) -> None:
        while self.global_step < total_steps:
            t0 = time.time()
            self.collect_rollout()
            losses = self.update()
            sps = self.rollout_len * self.env.n / (time.time() - t0)
            if curriculum is not None:
                ev = curriculum.maybe_eval(self.global_step, self.policy)
                if ev is not None:
                    losses.update({k: v for k, v in ev.items()
                                   if isinstance(v, (int, float))})
                    combined = float(np.nanmean([ev["eval_success_h"],
                                                 ev["eval_success_v"]]))
                    if combined >= self.best_eval:
                        self.best_eval = combined
                        self.save("best")
                    print(f"[{self.global_step:>9}] EVAL "
                          f"h={ev['eval_success_h']:.0%} v={ev['eval_success_v']:.0%} "
                          f"h_prob={ev['h_prob']:.2f} adapted={ev['adapted']}",
                          flush=True)
            summary = self.log(losses, sps)
            if self.update_idx % 10 == 0:
                print(f"[{self.global_step:>9}] " + " ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in summary.items() if k != "step"), flush=True)
            if self.update_idx % 50 == 0:
                self.save("latest")
        self.save("final")
