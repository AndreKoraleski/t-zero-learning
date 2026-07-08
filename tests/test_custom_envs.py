"""Custom environments: generic contract tests + per-env semantic tests.

Contract tests are auto-discovered from the gym registry (everything whose
entry point lives in ``envs.custom_envs``) — registering a new env opts it
in automatically.  Semantic tests encode what each env is *supposed* to do
(reward math, goal features, schedule switching); every new env should add
its own — use these as the template.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import envs.custom_envs  # noqa: F401 — registration side effects

CUSTOM_ENV_IDS = sorted(
    spec.id
    for spec in gym.registry.values()
    if str(spec.entry_point).startswith("envs.custom_envs")
)


# ---------------------------------------------------------------------------
# Generic contract (parametrized — covers new registrations automatically)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id", CUSTOM_ENV_IDS)
def test_env_contract(env_id):
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
    env.close()


@pytest.mark.parametrize("env_id", CUSTOM_ENV_IDS)
def test_env_reset_is_seed_deterministic(env_id):
    env1, env2 = gym.make(env_id), gym.make(env_id)
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)
    env1.close()
    env2.close()


# ---------------------------------------------------------------------------
# HalfCheetahVel-v1: reward = -|x_velocity - target_vel| + reward_ctrl
# ---------------------------------------------------------------------------

def test_halfcheetahvel_reward_math():
    env = gym.make("HalfCheetahVel-v1", target_vel=0.5)
    env.reset(seed=0)
    for _ in range(5):
        _, reward, _, _, info = env.step(env.action_space.sample())
        expected = -abs(info["x_velocity"] - 0.5) + info["reward_ctrl"]
        assert reward == pytest.approx(expected)
        assert info["reward_forward"] == pytest.approx(-abs(info["x_velocity"] - 0.5))
    env.close()


# ---------------------------------------------------------------------------
# AntDir-v1: reward_forward = v · û  (velocity projected on target direction)
# ---------------------------------------------------------------------------

def test_antdir_reward_math():
    env = gym.make("AntDir-v1", target_dir=[0.0, 1.0])
    env.reset(seed=0)
    for _ in range(5):
        _, reward, _, _, info = env.step(env.action_space.sample())
        expected_forward = info["y_velocity"]  # v · (0, 1)
        assert info["reward_forward"] == pytest.approx(expected_forward)
        expected = (
            expected_forward
            + info["reward_ctrl"]
            + info["reward_contact"]
            + info["reward_survive"]
        )
        assert reward == pytest.approx(expected)
    env.close()


# ---------------------------------------------------------------------------
# AntDirGoal-v1: appended goal features + direction schedule
# ---------------------------------------------------------------------------

BASE_ANT_OBS_DIM = gym.make("Ant-v5").observation_space.shape[0]


def test_antdir_goal_appends_8_features():
    env = gym.make("AntDirGoal-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (BASE_ANT_OBS_DIM + 8,)
    env.close()


def test_antdir_goal_dtheta_appends_9_features():
    env = gym.make("AntDirGoal-v1", obs_include_dtheta=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (BASE_ANT_OBS_DIM + 9,)
    env.close()


def test_antdir_goal_feature_values():
    """Goal block layout: [ux, uy, vx, vy, vx-ux, vy-uy, sin_err, cos_err]."""
    env = gym.make("AntDirGoal-v1")
    env.reset(seed=0)
    obs, _, _, _, info = env.step(env.action_space.sample())

    ux, uy = info["target_dir"]
    vx, vy = info["x_velocity"], info["y_velocity"]
    goal = obs[-8:]
    np.testing.assert_allclose(goal[:6], [ux, uy, vx, vy, vx - ux, vy - uy])
    # sin/cos of the heading error form a unit vector
    assert goal[6] ** 2 + goal[7] ** 2 == pytest.approx(1.0)
    env.close()


def test_antdir_goal_fixed_schedule_switches_direction():
    env = gym.make(
        "AntDirGoal-v1",
        schedule_mode="fixed",
        directions=[[1.0, 0.0], [0.0, 1.0]],
        switch_after_steps=[3],
    )
    _, info = env.reset(seed=0)
    np.testing.assert_allclose(info["target_dir"], [1.0, 0.0])

    seen = []
    for _ in range(6):
        _, _, _, _, info = env.step(env.action_space.sample())
        seen.append((info["direction_segment"], tuple(info["target_dir"])))

    # Steps 1-3 in segment 0, steps 4+ in segment 1
    assert [s for s, _ in seen] == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(seen[0][1], (1.0, 0.0))
    np.testing.assert_allclose(seen[-1][1], (0.0, 1.0))
    env.close()


def test_antdir_goal_random_two_samples_from_pool():
    env = gym.make(
        "AntDirGoal-v1",
        schedule_mode="random_two",
        switch_after_steps=None,
        directions=None,
        switch_step_range=(5, 10),
    )
    _, info = env.reset(seed=0)

    assert 5 <= info["sampled_switch_step"] <= 10
    d0, d1 = info["sampled_directions"]
    assert not np.allclose(d0, d1), "the two sampled directions must differ"
    env.close()


def test_antdir_goal_reward_math():
    env = gym.make("AntDirGoal-v1")
    env.reset(seed=0)
    for _ in range(5):
        _, reward, _, _, info = env.step(env.action_space.sample())
        ux, uy = info["target_dir"]
        expected_forward = info["x_velocity"] * ux + info["y_velocity"] * uy
        assert info["reward_forward"] == pytest.approx(expected_forward)
        expected = (
            expected_forward
            + info["reward_ctrl"]
            + info["reward_contact"]
            + info["reward_survive"]
        )
        assert reward == pytest.approx(expected)
    env.close()
