import gymnasium as gym
import numpy as np


class TargetVelRewardWrapper(gym.Wrapper):

    def __init__(self, env, target_vel=0.0):
        self.target_vel = np.float64(target_vel)
        super().__init__(env)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._get_rew(info)
        return observation, reward, terminated, truncated, info | reward_info
    
    def _get_rew(self, info):
        reward_forward = -1.0 * abs(info['x_velocity'] - self.target_vel)
        reward_ctrl = info['reward_ctrl']
        reward = reward_forward + reward_ctrl
        reward_info = dict(reward_forward=reward_forward,
                           reward_total=reward)
        return reward, reward_info

def make_target_vel_env(render_mode="rgb_array", **kwargs):
    base_env = gym.make("HalfCheetah-v5", render_mode=render_mode)  
    return TargetVelRewardWrapper(base_env, **kwargs)