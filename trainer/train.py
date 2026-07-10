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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_name = args.run or f"{cfg['stage']}_{time.strftime('%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    env = MinecraftVecEnv(
        curriculum=cfg["curriculum"],
        stage=cfg["stage"],
        seed=cfg.get("seed", 0),
        downsample=cfg.get("downsample", 2),
        frame_stack=cfg.get("frame_stack", 4),
        record_dir=str(run_dir / "episodes"),
    )
    trainer = PPOTrainer(env, run_dir, cfg["ppo"])
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
