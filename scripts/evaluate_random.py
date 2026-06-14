import argparse
import statistics
import sys
from pathlib import Path

import gymnasium as gym

# Allow running this file as: python scripts/evaluate_random.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.swarmball_env import SwarmBall  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate random agent baseline.")

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for evaluation episodes.",
    )

    parser.add_argument(
        "--goal-target",
        type=float,
        default=500.0,
        help="Distance from the initial goal object position that should be reached.",
    )

    parser.add_argument(
        "--enemy-acceleration",
        type=float,
        default=0.003,
        help="Enemy acceleration used during evaluation.",
    )

    parser.add_argument(
        "--enemy-max-speed",
        type=float,
        default=3.0,
        help="Enemy maximum speed used during evaluation.",
    )

    return parser.parse_args()


def make_env(args, seed: int):
    env = SwarmBall(
        number_of_clusters=3,
        number_of_bots_per_cluster=10,
        acc_factor=0.12,
        v_max=8,
        goal_target=args.goal_target,
        enemy_acceleration=args.enemy_acceleration,
        enemy_max_speed=args.enemy_max_speed,
    )

    # If the environment already returns Box observations, this wrapper is harmless.
    env = gym.wrappers.FlattenObservation(env)

    env.reset(seed=seed)

    return env


def run_episode(env, max_steps: int, seed: int):
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    episode_length = 0

    last_info = {}
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_length = step + 1
        last_info = info

        if terminated or truncated:
            break

    is_success = bool(last_info.get("is_success", False))
    enemy_caught = bool(last_info.get("enemy_caught", False))
    time_limit = bool(truncated and not terminated)

    return total_reward, episode_length, is_success, enemy_caught, time_limit


def main():
    args = parse_args()

    env = make_env(args=args, seed=args.seed)

    episode_rewards = []
    episode_lengths = []

    success_count = 0
    enemy_caught_count = 0
    time_limit_count = 0

    print("Evaluating random agent baseline...")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Seed: {args.seed}")
    print(f"Goal target: {args.goal_target}")
    print(f"Enemy acceleration: {args.enemy_acceleration}")
    print(f"Enemy max speed: {args.enemy_max_speed}")

    for episode in range(args.episodes):
        episode_seed = args.seed + episode

        (
            total_reward,
            episode_length,
            is_success,
            enemy_caught,
            time_limit,
        ) = run_episode(
            env=env,
            max_steps=args.max_steps,
            seed=episode_seed,
        )

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        if is_success:
            success_count += 1

        if enemy_caught:
            enemy_caught_count += 1

        if time_limit:
            time_limit_count += 1

        print(
            f"Episode {episode + 1}/{args.episodes}: "
            f"reward={total_reward:.2f}, "
            f"length={episode_length}, "
            f"success={is_success}, "
            f"enemy_caught={enemy_caught}, "
            f"time_limit={time_limit}"
        )

    env.close()

    print("\nRandom baseline summary:")
    print(f"Mean reward: {statistics.mean(episode_rewards):.2f}")
    print(f"Median reward: {statistics.median(episode_rewards):.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    print(f"Std reward: {statistics.stdev(episode_rewards):.2f}" if len(episode_rewards) > 1 else "Std reward: 0.00")
    print(f"Mean episode length: {statistics.mean(episode_lengths):.2f}")
    print(f"Success count: {success_count}/{args.episodes}")
    print(f"Enemy caught count: {enemy_caught_count}/{args.episodes}")
    print(f"Time limit count: {time_limit_count}/{args.episodes}")


if __name__ == "__main__":
    main()