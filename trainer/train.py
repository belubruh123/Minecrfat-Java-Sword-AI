"""Train entrypoint.

Usage: python train.py configs/stage1_aim.yaml [--steps N] [--run NAME]
The dev server must be running: DRLAGENT_TRAIN=1 ./gradlew runServer (in mod/)
"""

import argparse
import time
from pathlib import Path

import yaml

from drlagent.curriculum import RehearsalCurriculum
from drlagent.ppo import PPOTrainer
from drlagent.vec_env import MinecraftVecEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path")
    # Overrides so an orchestrator (scripts/train_all.sh) can rewire the
    # warm-start chain without editing YAMLs. "none" disables the config value
    # (e.g. train a fighter from scratch instead of resuming our lineage).
    ap.add_argument("--init-checkpoint", default=None,
                    help="override config init_checkpoint; 'none' disables it")
    ap.add_argument("--aim-checkpoint", default=None,
                    help="override config aim_checkpoint (frozen aim net)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_name = args.run or f"{cfg['stage']}_{time.strftime('%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    init_ckpt = cfg.get("init_checkpoint")
    if args.init_checkpoint is not None:
        init_ckpt = None if args.init_checkpoint.lower() == "none" else args.init_checkpoint
    aim_ckpt = args.aim_checkpoint or cfg.get("aim_checkpoint")

    env = MinecraftVecEnv(
        curriculum=cfg["curriculum"],
        stage=cfg["stage"],
        seed=cfg.get("seed", 0),
        downsample=cfg.get("downsample", 2),
        frame_stack=cfg.get("frame_stack", 4),
        record_dir=str(run_dir / "episodes"),
    )
    trainer = PPOTrainer(env, run_dir, cfg["ppo"], stage=cfg["stage"],
                         aim_ckpt=aim_ckpt,
                         swing_ckpt=cfg.get("swing_checkpoint"),
                         init_ckpt=init_ckpt)
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} at step {trainer.global_step}")

    curriculum = None
    if "curriculum_manager" in cfg:
        curriculum = RehearsalCurriculum(env, cfg["curriculum"], cfg["stage"],
                                         **cfg["curriculum_manager"])

    total = args.steps or cfg["total_steps"]
    print(f"run={run_name} arenas={env.n} obs={env.stack}x{env.h}x{env.w} "
          f"target={total} steps curriculum={'adaptive' if curriculum else 'fixed'}")
    try:
        trainer.train(total, curriculum=curriculum)
    finally:
        trainer.save("latest")
        env.close()


if __name__ == "__main__":
    main()
