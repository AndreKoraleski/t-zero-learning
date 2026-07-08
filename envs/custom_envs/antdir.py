"""
Ant direction: single fixed planar target direction; forward reward aligns velocity
with that direction (no extra goal features in observations).

"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.custom_envs.antdir_goal import _ANT_GOAL_KEYS


class TargetDirRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_dir):
        self.u = np.array(target_dir, dtype=np.float64)
        super().__init__(env)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._get_rew(info)
        return observation, reward, terminated, truncated, info | reward_info

    def _get_forward_reward(self, x_velocity, y_velocity):
        v = np.array([x_velocity, y_velocity])
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-8:
            return 0.0

        norm_u = np.linalg.norm(self.u)

        if norm_u < 1e-8:
            return 0.0

        v_normalized = v / norm_v
        u_normalized = self.u / norm_u
        return norm_v * np.dot(v_normalized, u_normalized)

    def _get_rew(self, info):
        reward_forward = self._get_forward_reward(info["x_velocity"], info["y_velocity"])
        reward_ctrl = info["reward_ctrl"]
        reward_contact = info["reward_contact"]
        reward_survive = info["reward_survive"]
        reward = reward_forward + reward_ctrl + reward_contact + reward_survive
        reward_info = dict(
            reward_forward=reward_forward,
            reward_total=reward,
        )
        return reward, reward_info

    def render(self):
        return self.env.render()


def make_target_dir_env(render_mode="rgb_array", target_dir=None, **kwargs):
    if target_dir is None:
        target_dir = [1.0, 1.0]
    ant_kwargs = {k: v for k, v in kwargs.items() if k not in _ANT_GOAL_KEYS}
    base_env = gym.make(
        "Ant-v5",
        render_mode=render_mode,
        healthy_z_range=(0.2, 0.9),
        **ant_kwargs,
    )
    return TargetDirRewardWrapper(base_env, target_dir)
