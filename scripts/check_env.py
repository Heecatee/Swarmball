import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

# Allow running this file as: python scripts/check_env.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.swarmball_env import SwarmBall  # noqa: E402


def main():
    env = SwarmBall(number_of_clusters=3)

    print("Checking raw SwarmBall environment...")
    check_env(env, warn=True)
    print("Raw environment is compatible with Stable-Baselines3.")

    flattened_env = gym.wrappers.FlattenObservation(env)

    print("Checking flattened SwarmBall environment...")
    check_env(flattened_env, warn=True)
    print("Flattened environment is compatible with Stable-Baselines3.")

    flattened_env.close()


if __name__ == "__main__":
    main()