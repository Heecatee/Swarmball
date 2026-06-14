import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Use dummy drivers for headless evaluation.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.swarmball_env import SwarmBall  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model without rendering.")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model without .zip extension.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed used for evaluation episodes.",
    )

    parser.add_argument(
        "--goal-target",
        type=float,
        default=500.0,
        help="Target X position used by the environment.",
    )

    parser.add_argument(
        "--enemy-max-speed",
        type=float,
        default=3.0,
        help="Maximum enemy speed.",
    )

    parser.add_argument(
        "--enemy-acceleration",
        type=float,
        default=0.003,
        help="Enemy acceleration.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Safety limit for one episode.",
    )

    return parser.parse_args()


def make_env(args):
    return SwarmBall(
        number_of_clusters=3,
        number_of_bots_per_cluster=10,
        goal_target=args.goal_target,
        enemy_max_speed=args.enemy_max_speed,
        enemy_acceleration=args.enemy_acceleration,
        render_mode=None,
    )


def get_result_label(terminated, truncated, info):
    if info.get("is_success"):
        return "SUCCESS"
    if info.get("enemy_caught"):
        return "FAILURE: enemy caught the object"
    if truncated:
        return "TIME LIMIT"
    if terminated:
        return "TERMINATED"
    return "UNKNOWN"


def main():
    args = parse_args()

    env = make_env(args)
    
    model_file = Path(args.model_path)
    if model_file.suffix != ".zip":
        model_file = model_file.with_suffix(".zip")

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. "
            "Pass --model-path without .zip, for example: "
            "runs/ppo_2m_seed_42/best_model/best_model"
        )

    model = PPO.load(args.model_path, device="cpu")

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    enemy_caught_count = 0
    time_limit_count = 0

    print("Evaluating PPO model...")
    print(f"Model path: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Evaluation seed: {args.seed}")
    print(f"Goal target: {args.goal_target}")
    print(f"Enemy max speed: {args.enemy_max_speed}")
    print(f"Enemy acceleration: {args.enemy_acceleration}")

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1

        if step >= args.max_steps and not terminated:
            truncated = True

        result = get_result_label(terminated, truncated, info)

        if info.get("is_success"):
            success_count += 1
        elif info.get("enemy_caught"):
            enemy_caught_count += 1
        elif truncated:
            time_limit_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        print(
            f"Episode {episode + 1:>3}: "
            f"reward={total_reward:>8.2f}, "
            f"steps={step:>4}, "
            f"result={result}"
        )

    env.close()

    print("\nEvaluation summary:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Median reward: {np.median(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"Success count: {success_count}/{args.episodes}")
    print(f"Enemy caught count: {enemy_caught_count}/{args.episodes}")
    print(f"Time limit count: {time_limit_count}/{args.episodes}")


if __name__ == "__main__":
    main()