import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Use headless PyGame drivers during training.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Allow running this file as: python scripts/train_ppo.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from environment.swarmball_env import SwarmBall
except ImportError:
    from swarmball_env import SwarmBall


class TrainingMetricsCallback(BaseCallback):
    """Print basic training metrics during PPO training."""

    def __init__(self, print_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print < self.print_freq:
            return True

        self._last_print = self.num_timesteps

        if len(self.model.ep_info_buffer) == 0:
            return True

        episode_rewards = [episode_info["r"] for episode_info in self.model.ep_info_buffer]
        episode_lengths = [episode_info["l"] for episode_info in self.model.ep_info_buffer]

        print(
            f"[{self.num_timesteps:>8} steps] "
            f"mean_reward={np.mean(episode_rewards):>8.1f}  "
            f"mean_len={np.mean(episode_lengths):>6.0f}  "
            f"best_reward={np.max(episode_rewards):>8.1f}"
        )

        return True


def make_env(
    rank: int,
    seed: int,
    goal_target: float,
    enemy_acceleration: float,
    enemy_max_speed: float,
    acc_factor: float,
    v_max: float,
    monitor_dir: Path | None = None,
):
    """Create one monitored SwarmBall environment instance."""

    def _init():
        env = SwarmBall(
            number_of_clusters=3,
            number_of_bots_per_cluster=10,
            acc_factor=acc_factor,
            v_max=v_max,
            goal_target=goal_target,
            enemy_acceleration=enemy_acceleration,
            enemy_max_speed=enemy_max_speed,
        )

        monitor_file = None
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            monitor_file = str(monitor_dir / f"env_{rank}")

        env = Monitor(env, filename=monitor_file)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_vector_env(args, run_dir: Path):
    """Create the vectorized training environment."""

    monitor_dir = run_dir / "monitor"

    env_factories = [
        make_env(
            rank=rank,
            seed=args.seed,
            goal_target=args.goal_target,
            enemy_acceleration=args.enemy_acceleration,
            enemy_max_speed=args.enemy_max_speed,
            monitor_dir=monitor_dir,
            acc_factor=args.acc_factor,
            v_max=args.v_max,
        )
        for rank in range(args.n_envs)
    ]

    if args.n_envs > 1:
        return SubprocVecEnv(env_factories)

    return DummyVecEnv(env_factories)


def create_eval_env(args, run_dir: Path):
    """Create a separate environment for periodic model evaluation."""

    return DummyVecEnv(
        [
            make_env(
                rank=0,
                seed=args.seed + 10_000,
                goal_target=args.goal_target,
                enemy_acceleration=args.enemy_acceleration,
                enemy_max_speed=args.enemy_max_speed,
                acc_factor=args.acc_factor,
                v_max=args.v_max,
                monitor_dir=run_dir / "eval_monitor",
            )
        ]
    )


def compute_batch_size(n_steps: int, n_envs: int) -> int:
    """Choose the largest common PPO batch size for the rollout buffer."""

    total_steps_per_update = n_steps * n_envs

    for batch_size in [512, 256, 128, 64, 32]:
        if total_steps_per_update % batch_size == 0:
            return batch_size

    return 64


def create_model(args, vec_env, tensorboard_dir: Path, batch_size: int, n_steps: int) -> PPO:
    """Create a new PPO model."""

    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=args.device,
        learning_rate=args.lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
    )


def train(args):
    print("=" * 80)
    print("SwarmBall PPO Training")
    print("=" * 80)

    run_dir = Path("runs") / args.run_name
    model_path = run_dir / "model"
    best_model_dir = run_dir / "best_model"
    checkpoint_dir = run_dir / "checkpoints"
    eval_log_dir = run_dir / "eval_logs"
    tensorboard_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing_model = model_path.with_suffix(".zip")
    if existing_model.exists() and not args.continue_training:
        raise FileExistsError(
            f"Model already exists: {existing_model}. "
            f"Use --continue-training to continue it or choose a different --run-name."
        )
        
    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Training seed: {args.seed}")
    print(f"Target position: {args.goal_target}")
    print(f"Enemy acceleration: {args.enemy_acceleration}")
    print(f"Enemy max speed: {args.enemy_max_speed}")
    print(f"Parallel environments: {args.n_envs}")

    vec_env = create_vector_env(args, run_dir)
    eval_env = create_eval_env(args, run_dir)

    n_steps = args.n_steps
    batch_size = compute_batch_size(n_steps=n_steps, n_envs=args.n_envs)
    print(f"n_steps={n_steps}, n_envs={args.n_envs}, batch_size={batch_size}")

    if args.continue_training:
        if not existing_model.exists():
            raise FileNotFoundError(
                f"Cannot continue training because model does not exist: {existing_model}"
            )

        print(f"Loading existing model from {existing_model}...")
        model = PPO.load(
            str(model_path),
            env=vec_env,
            device=args.device,
            learning_rate=args.lr,
        )
        reset_num_timesteps = False
    else:
        print("Creating a new PPO model...")
        model = create_model(
            args=args,
            vec_env=vec_env,
            tensorboard_dir=tensorboard_dir,
            batch_size=batch_size,
            n_steps=n_steps,
        )
        reset_num_timesteps = True

    print("\nPolicy architecture:")
    print(model.policy)

    # Callback frequencies are counted in VecEnv step calls.
    checkpoint_save_freq = max(args.timesteps // (20 * max(args.n_envs, 1)), 1)
    eval_freq = max(args.timesteps // (10 * max(args.n_envs, 1)), 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="swarmball_ppo",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    metrics_callback = TrainingMetricsCallback(
        print_freq=max(args.timesteps // 100, 2_000)
    )

    print(f"\nStarting training: {args.timesteps:,} timesteps on '{args.device}'")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback, metrics_callback],
            progress_bar=False,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving the current model state.")

    model.save(str(model_path))
    print(f"\nFinal model saved to: {model_path}.zip")
    print(f"Best model directory: {best_model_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard directory: {tensorboard_dir}")

    print(f"\n--- Final evaluation: {args.eval_episodes} episodes ---")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    print("\nUseful commands:")
    print(
        f"python scripts/evaluate_model.py "
        f"--model-path {best_model_dir / 'best_model'} "
        f"--goal-target {args.goal_target} "
        f"--enemy-acceleration {args.enemy_acceleration} "
        f"--enemy-max-speed {args.enemy_max_speed}"
    )
    print(
        f"python scripts/watch_model.py "
        f"--model-path {best_model_dir / 'best_model'} "
        f"--goal-target {args.goal_target} "
        f"--enemy-acceleration {args.enemy_acceleration} "
        f"--enemy-max-speed {args.enemy_max_speed}"
    )
    
    vec_env.close()
    eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO model for SwarmBall.")

    parser.add_argument(
        "--run-name",
        type=str,
        default="ppo_2m_seed_42",
        help="Name of the training run. Outputs are saved to runs/<run-name>/.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Number of training timesteps.",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed used for environment resets and PPO initialization.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help="Discount factor.",
    )

    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.005,
        help="Entropy coefficient used for exploration.",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of rollout steps per environment before each PPO update.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, or cuda.",
    )

    parser.add_argument(
        "--goal-target",
        type=float,
        default=500.0,
        help="Target X position that the goal object should reach.",
    )

    parser.add_argument(
        "--enemy-acceleration",
        type=float,
        default=0.003,
        help="Enemy acceleration used in both training and evaluation environments.",
    )

    parser.add_argument(
        "--enemy-max-speed",
        type=float,
        default=3.0,
        help="Enemy maximum speed used in both training and evaluation environments.",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes used by EvalCallback and final evaluation.",
    )

    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue training from runs/<run-name>/model.zip instead of starting a new model.",
    )
    
    parser.add_argument(
        "--acc-factor",
        type=float,
        default=0.12,
        help="Acceleration factor for threshold movement.",
    )

    parser.add_argument(
        "--v-max",
        type=float,
        default=8.0,
        help="Maximum threshold velocity.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())