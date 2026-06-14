import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .simulation.simulation import SwarmBallSimulation
except ImportError:
    from simulation.simulation import SwarmBallSimulation


class SwarmBall(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        acc_factor=0.12,        
        number_of_clusters=3,
        v_max=8,                
        number_of_bots_per_cluster=10,
        goal_target=300.0,
        render_mode=None,
        **kwargs,
    ):
        super().__init__()

        self.sim = SwarmBallSimulation(
            number_of_clusters=number_of_clusters,
            number_of_bots_per_cluster=number_of_bots_per_cluster,
            **kwargs,
        )
        self.cluster_count = number_of_clusters
        self.bots_per_cluster = number_of_bots_per_cluster
        self.thresh_vel = np.zeros(number_of_clusters, dtype=np.float32)
        self.v_max = v_max
        self.acc_factor = acc_factor
        self.render_mode = render_mode
        self.goal_target = float(goal_target)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(number_of_clusters,), dtype=np.float32
        )

        # Obserwacje: [goal_pos, goal_vel, enemy_gap, enemy_speed, dist_to_target,
        #              (thresh_rel, bots_rel, n_bots, thresh_v) * N]
        obs_size = 5 + number_of_clusters * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self._prev_goal_pos = 0.0
        self._step_count = 0
        self._initial_goal_pos = 0.0

    # ------------------------------------------------------------------
    # Obserwacja
    # ------------------------------------------------------------------

    def _get_obs(self):
        goal_pos = float(self.sim._goal_object.body.position[0])
        goal_vel = float(self.sim._goal_object.body.velocity[0])

        enemy_gap   = goal_pos - float(self.sim._enemy_position)
        enemy_speed = float(self.sim._enemy_speed)

        dist_to_target = (self.goal_target - goal_pos) / max(self.goal_target, 1.0)

        parts = [
            goal_pos / 1000.0,
            goal_vel / 50.0,
            enemy_gap / 1000.0,
            enemy_speed / 10.0,
            dist_to_target,
        ]

        thresh_positions = self.sim.threshold_positions()
        for i, cluster in enumerate(self.sim._clusters):
            thresh_rel = (thresh_positions[i] - goal_pos) / 500.0

            if len(cluster.bots) > 0:
                bots_x = [b.body.position[0] for b in cluster.bots]
                bots_mean = float(np.mean(bots_x))
            else:
                bots_mean = goal_pos
            bots_rel = (bots_mean - goal_pos) / 500.0

            n_bots_norm   = len(cluster.bots) / float(self.bots_per_cluster)
            thresh_v_norm = self.thresh_vel[i] / self.v_max

            parts.extend([thresh_rel, bots_rel, n_bots_norm, thresh_v_norm])

        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        goal_pos = float(self.sim._goal_object.body.position[0])
        goal_vel = float(self.sim._goal_object.body.velocity[0])
        thresh_positions = self.sim.threshold_positions()

        # 1. Postęp kwadratu w prawo — główna nagroda
        progress = goal_pos - self._prev_goal_pos
        reward_progress = progress * 5.0

        # 2. Nagroda za to że threshold jest TUŻ ZA kwadratem (nie za daleko, nie za blisko)
        #    Optymalny punkt: threshold ~30-80 jednostek za kwadratem żeby boty go pchały
        total_bots = 0
        bots_pushing = 0
        for cluster in self.sim._clusters:
            for bot in cluster.bots:
                total_bots += 1
                if bot.body.position[0] < goal_pos + 10:  # bot jest za lub tuż przy kwadracie
                    bots_pushing += 1

        if total_bots > 0:
            push_ratio = bots_pushing / total_bots
            reward_pushing = push_ratio * 1.5
        else:
            reward_pushing = 0.0

        # 3. Nagroda za threshold blisko kwadratu (od tyłu, +20..+100 za goal)
        #    Threshold POWINIEN być trochę za kwadratem żeby boty mogły go dosięgnąć
        mean_thresh = float(np.mean(thresh_positions))
        # Idealny offset: threshold 40 jednostek ZA kwadratem (po lewej)
        ideal_thresh_pos = goal_pos - 40.0
        thresh_offset_err = abs(mean_thresh - ideal_thresh_pos)
        # Gaussowska nagroda - max gdy threshold dokładnie 40 jednostek za kwadratem
        reward_thresh_pos = np.exp(-thresh_offset_err / 150.0) * 1.0

        # 4. Nagroda za prędkość kwadratu w prawo
        reward_velocity = max(0.0, goal_vel) * 0.15

        # 5. Dystans do wroga — lekka nagroda za ucieczkę
        enemy_gap = goal_pos - float(self.sim._enemy_position)
        reward_enemy_gap = np.tanh(enemy_gap / 300.0) * 0.2
        if enemy_gap < 100.0:
            reward_enemy_gap -= (100.0 - enemy_gap) * 0.005
        if enemy_gap <= 0.0:
            reward_enemy_gap -= 5.0

        # 6. Kara za stracone boty
        expected_bots = self.cluster_count * self.bots_per_cluster
        reward_bots = -0.3 * (expected_bots - total_bots) / max(expected_bots, 1)

        # 7. Postęp % do mety
        if self._initial_goal_pos < self.goal_target:
            total_dist = self.goal_target - self._initial_goal_pos
            progress_pct = max(0.0, (goal_pos - self._initial_goal_pos) / total_dist)
            reward_target_progress = progress_pct * 0.5
        else:
            reward_target_progress = 0.0

        total_reward = (
            reward_progress
            + reward_pushing
            + reward_thresh_pos
            + reward_velocity
            + reward_enemy_gap
            + reward_bots
            + reward_target_progress
        )
        return float(total_reward)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(self, action):
        # ZMIANA: bardziej responsywna akceleracja (mniej smoothing = szybsza reakcja)
        self.thresh_vel = (
            self.thresh_vel * 0.4                          # było 0.6 — mniej inercji
            + (action - 0.5) * 2.0 * self.acc_factor * self.v_max
        )
        self.thresh_vel = np.clip(self.thresh_vel, -self.v_max, self.v_max)

        thresh_positions = self.sim.threshold_positions()
        for i in range(self.cluster_count):
            new_pos = thresh_positions[i] + self.thresh_vel[i]
            self.sim.update_thresholds_position(i, new_pos)

        self.sim.step()
        self._step_count += 1

        obs    = self._get_obs()
        reward = self._compute_reward()

        goal_pos     = float(self.sim._goal_object.body.position[0])
        enemy_caught = bool(self.sim._enemy_position >= goal_pos)
        goal_reached = bool(goal_pos >= self.goal_target)
        terminated   = enemy_caught or goal_reached

        truncated = self._step_count >= 3000

        if goal_reached and not enemy_caught:
            speed_bonus = max(0.0, 1.0 - self._step_count / 3000.0)
            reward += 2000.0 + speed_bonus * 1000.0
        if enemy_caught:
            reward -= 500.0

        self._prev_goal_pos = goal_pos

        info = {
            "is_success":        goal_reached and not enemy_caught,
            "enemy_caught":      enemy_caught,
            "goal_progress_pct": max(0.0, (goal_pos - self._initial_goal_pos) /
                                     max(self.goal_target - self._initial_goal_pos, 1.0)),
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.thresh_vel     = np.zeros(self.cluster_count, dtype=np.float32)
        self._step_count    = 0
        self.sim.reset()
        self._prev_goal_pos    = float(self.sim._goal_object.body.position[0])
        self._initial_goal_pos = self._prev_goal_pos

        obs  = self._get_obs()
        info = {}
        return obs, info

    def render(self):
        if self.render_mode == "human":
            self.sim.redraw(goal_target=self.goal_target)
        elif self.render_mode == "rgb_array":
            return self.sim.space_near_goal_object()

    def close(self):
        pass