import argparse
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.swarmball_env import SwarmBall  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained PPO model with PyGame rendering.")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model without .zip extension.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to render.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed used for rendered episodes.",
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
        "--delay",
        type=float,
        default=0.002,
        help="Delay between rendered frames in seconds.",
    )

    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help="Continue with the next episode after termination.",
    )

    return parser.parse_args()


def make_env(args):
    return SwarmBall(
        number_of_clusters=3,
        number_of_bots_per_cluster=10,
        goal_target=args.goal_target,
        enemy_max_speed=args.enemy_max_speed,
        enemy_acceleration=args.enemy_acceleration,
        render_mode="human",
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
    model = PPO.load(args.model_path, device="cpu")

    print("Watching PPO model...")
    print(f"Model path: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Goal target: {args.goal_target}")
    print(f"Enemy max speed: {args.enemy_max_speed}")
    print(f"Enemy acceleration: {args.enemy_acceleration}")

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            env.render()
            time.sleep(args.delay)

        result = get_result_label(terminated, truncated, info)

        print(
            f"Episode {episode + 1:>3}: "
            f"reward={total_reward:>8.2f}, "
            f"steps={step:>4}, "
            f"result={result}"
        )

        if not args.reset_on_done:
            break

    env.close()


if __name__ == "__main__":
    main()