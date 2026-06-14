import os
import sys

import numpy as np

# Add project root to Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.swarmball_env import SwarmBall


def test_reset_returns_valid_observation():
    env = SwarmBall()

    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    env.close()


def test_step_returns_valid_values():
    env = SwarmBall()

    env.reset(seed=42)
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_step_info_contains_evaluation_fields():
    env = SwarmBall()

    env.reset(seed=42)
    action = env.action_space.sample()

    _, _, _, _, info = env.step(action)

    assert "is_success" in info
    assert "enemy_caught" in info
    assert "goal_progress_pct" in info

    env.close()