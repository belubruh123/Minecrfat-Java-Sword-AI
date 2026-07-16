"""Online fine-tuning from live in-game play — "pilot mode, but it learns."

Runs the SAME PPO as headless training, but the environment is a single live
player driven through the client pilot bridge, and the reward is computed
client-side in the mod (mirroring Arena.stepCombo). Only the combat head
(Fighter2Policy) trains; the aim net stays frozen — exactly the split the user
asked for.

Usage (from trainer/):
    python pilot_train.py [--resume ../models/fighter2.pt] [--aim ../models/aim.pt]
                          [--port 36566] [--steps N] [--run NAME] [--no-ship]

Start this FIRST, then in a Minecraft client (with the drlagent mod) load a
world with an opponent — a real player, or the TheoBaldTheBird HeroBot bot if
you have it for this MC version — target it, and press the pilot key (G). The
policy learns from every exchange. Press G again (or Ctrl+C here) to stop; the
improved fighter is shipped to models/fighter2.pt (previous one backed up).

Because live play is real-time and single-stream (~thousands x slower than the
8-arena headless sim), this is for FINE-TUNING a trained fighter, not training
from scratch.
"""

import argparse
import shutil
import socket
import time
from pathlib import Path

import torch

from drlagent.pilot_env import PilotEnv
from drlagent.ppo import PPOTrainer

MODELS = Path(__file__).resolve().parent.parent / "models"

# Fine-tune PPO settings (mirror configs/stage8_theobald.yaml): low LR and a
# tiny entropy bonus so a converged fighter sharpens instead of drifting apart.
DEFAULT_PPO = {
    "lr": 1.0e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "rollout_len": 128,
    "num_minibatches": 2,
    "update_epochs": 2,
    "clip_coef": 0.2,
    "ent_coef": 0.0007,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "reward_scale": 1.0,
    "torch_threads": 2,
}


def resume_into(trainer: PPOTrainer, path: Path) -> None:
    """Load policy (+ optimizer if present) from a shipped checkpoint. Tolerant
    of checkpoints saved without optimizer state (warm-start fallback)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trainer.policy.load_state_dict(ckpt["policy"])
    if "optimizer" in ckpt:
        try:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            print("  (optimizer state incompatible — starting Adam fresh)")
    print(f"resumed fighter from {path} "
          f"(trained {ckpt.get('step', '?')} steps, stage {ckpt.get('stage', '?')})")


def ship(trainer: PPOTrainer) -> None:
    """Copy the peak (or final) checkpoint to models/fighter2.pt, backing up the
    previous one — same convention as scripts/train_all.sh."""
    trainer.save("final")
    src = trainer.run_dir / "best.pt"
    if not src.exists():
        src = trainer.run_dir / "final.pt"
    dst = MODELS / "fighter2.pt"
    if dst.exists():
        shutil.copy2(dst, MODELS / "fighter2_prev.pt")
    shutil.copy2(src, dst)
    print(f"shipped {src.name} -> {dst} (previous -> fighter2_prev.pt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aim", default=str(MODELS / "aim.pt"))
    ap.add_argument("--resume", default=str(MODELS / "fighter2.pt"),
                    help="fighter2 checkpoint to fine-tune")
    ap.add_argument("--port", type=int, default=36566)
    ap.add_argument("--steps", type=int, default=1_000_000,
                    help="max env steps before auto-stop (usually you stop earlier)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--lr", type=float, default=DEFAULT_PPO["lr"])
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--no-ship", action="store_true",
                    help="don't copy the result into models/fighter2.pt on exit")
    args = ap.parse_args()

    if not Path(args.aim).exists():
        raise SystemExit(f"aim checkpoint not found: {args.aim} (train stage1/2 first)")
    if not Path(args.resume).exists():
        raise SystemExit(f"fighter checkpoint not found: {args.resume} "
                         f"(train the fighter first, or copy from a backup)")

    ppo = dict(DEFAULT_PPO, lr=args.lr)
    run_name = args.run or f"pilot_finetune_{time.strftime('%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    srv = socket.create_server(("127.0.0.1", args.port))
    print(f"pilot-train server listening on 127.0.0.1:{args.port}")
    print("  -> in Minecraft: load a world with an opponent, target it, press G")
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    env = PilotEnv(conn, downsample=args.downsample, frame_stack=args.frame_stack,
                   curriculum={"allow_jump": 0})
    print(f"client connected: {env.width}x{env.height} obs -> {env.w}x{env.h}, "
          f"training fighter2 (aim frozen)")

    trainer = PPOTrainer(env, run_dir, ppo, stage="fighter2", aim_ckpt=args.aim)
    resume_into(trainer, Path(args.resume))
    total = trainer.global_step + args.steps
    print(f"run={run_name} — learning from live play; press G (or Ctrl+C) to stop")
    try:
        trainer.train(total)
    except (ConnectionError, KeyboardInterrupt) as e:
        print(f"\nstopped: {type(e).__name__} {e}")
    finally:
        if not args.no_ship:
            ship(trainer)
        env.close()
        srv.close()


if __name__ == "__main__":
    main()
