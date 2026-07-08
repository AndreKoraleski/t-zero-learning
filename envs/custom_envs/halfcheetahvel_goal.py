"""
HalfCheetahVel with a multi-velocity pool and mid-episode velocity switching.

Appends the current target velocity to the observation, enabling the network to
condition its behaviour on the active velocity. Used as a 1D continuous task
manifold analogue of AntDirGoal (2D direction ring) for the velocity-manifold
neuron-sharing experiment.

Schedule modes:
  - fixed: use explicit ``velocities`` and ``switch_after_steps`` every episode.
  - random_two: each reset samples two distinct velocities from ``velocity_pool``
    and a switch step uniformly from ``switch_step_range`` (inclusive).

Switch semantics are identical to AntDirGoalWrapper: after
``sum(switch_after_steps[:k])`` completed steps, the active velocity advances to
``velocities[k+1]``.

"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

_HALFCHEETAHVEL_GOAL_KEYS = frozenset(
    {
        "schedule_mode",
        "velocities",
        "switch_after_steps",
        "velocity_pool",
        "switch_step_range",
        "upright_reward_weight",
    }
)


class HalfCheetahVelGoalWrapper(gym.Wrapper):
    """
    Wraps HalfCheetah-v5 with a velocity pool and mid-episode velocity
    switching. Appends ``[target_vel]`` (1 dim) to the observation.

    Parameters
    ----------
    env : gymnasium.Env
        Underlying environment (expected: ``HalfCheetah-v5`` with 1-D Box obs).
    schedule_mode : str, default ``"fixed"``
        ``"fixed"``: use ``velocities`` and ``switch_after_steps`` every episode.
        ``"random_two"``: each reset samples two distinct velocities from
        ``velocity_pool`` and a switch step from ``switch_step_range``.
    velocities : list of float | None
        Used when ``schedule_mode="fixed"``. Target velocities per segment.
        Default ``[1.0]``. Length ``n`` must match
        ``len(switch_after_steps) == n - 1``.
    switch_after_steps : list of int | None
        Used when ``schedule_mode="fixed"``. Positive segment lengths in env
        steps. Default ``[]`` (single segment).
    velocity_pool : list of float | None
        Candidate velocities for ``schedule_mode="random_two"``.
        Default: ``[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]``.
    switch_step_range : (int, int) | None
        Inclusive integer range for the first-segment length in
        ``"random_two"`` mode. Default ``(200, 800)``.
    """

    def __init__(
        self,
        env,
        schedule_mode: str = "fixed",
        velocities: list[float] | None = None,
        switch_after_steps: list[int] | None = None,
        velocity_pool: list[float] | None = None,
        switch_step_range: tuple[int, int] | list[int] | None = None,
        upright_reward_weight: float = 0.5,
    ):
        super().__init__(env)
        self.schedule_mode = schedule_mode

        if velocity_pool is None:
            self._velocity_pool = np.array(
                [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype=np.float64
            )
        else:
            self._velocity_pool = np.array(velocity_pool, dtype=np.float64)
            if self._velocity_pool.ndim != 1:
                raise ValueError("velocity_pool must be a 1-D list of floats")

        if switch_step_range is None:
            self._switch_step_range = (200, 800)
        else:
            lo, hi = switch_step_range
            self._switch_step_range = (int(lo), int(hi))

        self._fixed_velocities: np.ndarray | None = None
        self._fixed_switch_after: np.ndarray | None = None

        if schedule_mode == "fixed":
            if velocities is None:
                velocities = [1.0]
            vels = np.array(velocities, dtype=np.float64)
            if vels.ndim != 1:
                raise ValueError("velocities must be a 1-D list of floats")
            if switch_after_steps is None:
                switch_after_steps = [] if len(vels) == 1 else [500]
            sw = np.array(switch_after_steps, dtype=np.int64)
            exp_sw = max(0, len(vels) - 1)
            if sw.size != exp_sw:
                raise ValueError(
                    "switch_after_steps must have length len(velocities) - 1"
                )
            if exp_sw > 0 and np.any(sw <= 0):
                raise ValueError("switch_after_steps entries must be positive")
            self._fixed_velocities = vels
            self._fixed_switch_after = sw

        elif schedule_mode == "random_two":
            if len(self._velocity_pool) < 2:
                raise ValueError(
                    "random_two needs at least two velocities in velocity_pool"
                )
        else:
            raise ValueError("schedule_mode must be 'fixed' or 'random_two'")

        # Extend observation space by 1 dim for the target velocity.
        base_low = env.observation_space.low
        base_high = env.observation_space.high
        if env.observation_space.shape != (len(base_low),):
            raise ValueError(
                "HalfCheetahVelGoalWrapper expects 1-D Box observations"
            )
        low = np.concatenate([base_low, [-np.inf]])
        high = np.concatenate([base_high, [np.inf]])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64
        )

        self._upright_reward_weight = float(upright_reward_weight)

        self._steps_done: int = 0
        self._velocities: np.ndarray | None = self._fixed_velocities
        self._switch_after_steps: np.ndarray = (
            self._fixed_switch_after
            if self._fixed_switch_after is not None
            else np.zeros(0, dtype=np.int64)
        )
        self.target_vel: float = 1.0
        self._episode_switch_step: int | None = None
        self._episode_velocities_sample: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _segment_index(self) -> int:
        if self._velocities is None or len(self._velocities) == 0:
            return 0
        if len(self._switch_after_steps) == 0:
            return 0
        cum = np.cumsum(self._switch_after_steps)
        idx = int(np.searchsorted(cum, self._steps_done, side="left"))
        return min(idx, len(self._velocities) - 1)

    def _prepare_schedule(self) -> None:
        if self.schedule_mode == "fixed":
            self._velocities = self._fixed_velocities
            self._switch_after_steps = self._fixed_switch_after
            self._episode_switch_step = None
            self._episode_velocities_sample = None
            return

        rng = self.np_random
        lo, hi = self._switch_step_range
        switch_step = int(rng.integers(lo, hi + 1))
        idx = rng.choice(len(self._velocity_pool), size=2, replace=False)
        v0 = float(self._velocity_pool[int(idx[0])])
        v1 = float(self._velocity_pool[int(idx[1])])
        self._velocities = np.array([v0, v1], dtype=np.float64)
        self._switch_after_steps = np.array([switch_step], dtype=np.int64)
        self._episode_switch_step = switch_step
        self._episode_velocities_sample = self._velocities.copy()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray, info: dict) -> tuple[float, dict]:
        reward_forward = -1.0 * abs(float(info["x_velocity"]) - self.target_vel)
        reward_ctrl = float(info["reward_ctrl"])
        # obs[1] is rooty: torso rotation angle. cos(rooty)=1 upright, -1 flipped.
        rooty = float(obs[1])
        reward_upright = self._upright_reward_weight * float(np.cos(rooty))
        reward = reward_forward + reward_ctrl + reward_upright
        return reward, dict(
            reward_forward=reward_forward,
            reward_upright=reward_upright,
            reward_total=reward,
        )

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, [self.target_vel]], axis=-1)

    # ------------------------------------------------------------------
    # Schedule info for the info dict
    # ------------------------------------------------------------------

    def _schedule_info(self, segment_idx: int) -> dict:
        out: dict = dict(
            target_vel=self.target_vel,
            velocity_segment=segment_idx,
        )
        if self._episode_switch_step is not None:
            out["sampled_switch_step"] = self._episode_switch_step
        if self._episode_velocities_sample is not None:
            out["sampled_velocities"] = self._episode_velocities_sample.copy()
        return out

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._steps_done = 0
        self._prepare_schedule()
        self.target_vel = float(self._velocities[0])
        obs = self._augment_obs(obs)
        info = dict(info)
        info.update(self._schedule_info(0))
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._steps_done += 1
        seg = self._segment_index()
        self.target_vel = float(self._velocities[seg])
        reward, reward_info = self._compute_reward(obs, info)
        info = dict(info)
        info.update(reward_info)
        info.update(self._schedule_info(seg))
        obs = self._augment_obs(obs)
        return obs, reward, terminated, truncated, info


def make_halfcheetahvel_goal_env(render_mode: str = "rgb_array", **kwargs):
    hc_kwargs = {k: v for k, v in kwargs.items() if k not in _HALFCHEETAHVEL_GOAL_KEYS}
    goal_kwargs = {k: v for k, v in kwargs.items() if k in _HALFCHEETAHVEL_GOAL_KEYS}
    base_env = gym.make("HalfCheetah-v5", render_mode=render_mode, **hc_kwargs)
    return HalfCheetahVelGoalWrapper(base_env, **goal_kwargs)
