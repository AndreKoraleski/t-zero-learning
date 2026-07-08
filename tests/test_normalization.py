"""Agent-owned observation normalization: math, state_dict travel, agent wiring.

The normalizer replaces ``gym.wrappers.NormalizeObservation``; the tests pin
the properties that matter for training/eval correctness:

- running statistics match batch mean/var over everything seen so far;
- stats live in the agent's ``state_dict`` (checkpoints carry them for free);
- ``use_obs_norm=False`` is a strict no-op (and keeps the legacy state_dict
  format, so pre-normalization models still load);
- normalized outputs are clipped to ±10 (CleanRL's post-normalization clip);
- evaluation-style use (no ``update_norm``) leaves the stats frozen.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from networks import ContinuousActorCritic, ObsNormalizer


def _envs(env_id="Pendulum-v1", n=2):
    return gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n)])


# ---------------------------------------------------------------------------
# ObsNormalizer math
# ---------------------------------------------------------------------------

def test_running_stats_match_numpy():
    torch.manual_seed(0)
    norm = ObsNormalizer(obs_dim=3)
    batches = [torch.randn(16, 3) * 5.0 + 2.0 for _ in range(10)]
    for b in batches:
        norm.update(b)

    all_x = torch.cat(batches).numpy()
    np.testing.assert_allclose(norm.running.mean.numpy(), all_x.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(norm.running.var.numpy(), all_x.var(axis=0), rtol=1e-3)


def test_normalize_centers_and_scales():
    torch.manual_seed(0)
    norm = ObsNormalizer(obs_dim=4)
    x = torch.randn(1000, 4) * 3.0 - 7.0
    norm.update(x)

    y = norm.normalize(x)
    assert y.mean(dim=0).abs().max() < 0.1
    assert (y.std(dim=0) - 1.0).abs().max() < 0.1


def test_normalized_output_is_clipped():
    norm = ObsNormalizer(obs_dim=2)
    norm.update(torch.zeros(100, 2))  # mean 0, var ~0 -> huge normalized values
    y = norm.normalize(torch.full((1, 2), 1e6))
    assert torch.all(y <= 10.0) and torch.all(y >= -10.0)


def test_wrong_obs_dim_rejected():
    norm = ObsNormalizer(obs_dim=3)
    with pytest.raises(ValueError, match="obs_dim"):
        norm.update(torch.zeros(4, 5))
    with pytest.raises(ValueError, match="obs_dim"):
        norm.normalize(torch.zeros(4, 5))


def test_stats_travel_through_state_dict():
    torch.manual_seed(0)
    src = ObsNormalizer(obs_dim=3)
    src.update(torch.randn(64, 3) * 4.0 + 1.0)

    dst = ObsNormalizer(obs_dim=3)
    dst.load_state_dict(src.state_dict())

    x = torch.randn(8, 3)
    torch.testing.assert_close(dst.normalize(x), src.normalize(x))


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

def test_disabled_norm_is_identity_and_keeps_legacy_state_dict():
    envs = _envs()
    agent = ContinuousActorCritic(envs)  # use_obs_norm defaults to False

    x = torch.randn(5, envs.single_observation_space.shape[0])
    agent.update_norm(x)  # must be a no-op, not an error
    assert agent.normalize_obs(x) is x

    # No normalizer keys: pre-normalization checkpoints stay loadable.
    assert not any("obs_normalizer" in k for k in agent.state_dict())
    envs.close()


def test_enabled_norm_normalizes_inputs():
    torch.manual_seed(0)
    envs = _envs()
    agent = ContinuousActorCritic(envs, use_obs_norm=True)
    obs_dim = envs.single_observation_space.shape[0]

    x = torch.randn(256, obs_dim) * 50.0 + 100.0  # far outside N(0, 1)
    agent.update_norm(x)

    y = agent.normalize_obs(x)
    assert y.mean(dim=0).abs().max() < 0.5
    # get_action_and_value on raw vs pre-normalized input must agree.
    _, _, _, v_raw = agent.get_action_and_value(x, action=torch.zeros(256, envs.single_action_space.shape[0]))
    _, _, _, v_norm = agent.get_action_and_value(
        y, action=torch.zeros(256, envs.single_action_space.shape[0]), input_is_normalized=True
    )
    torch.testing.assert_close(v_raw, v_norm)
    envs.close()


def test_norm_stats_saved_and_restored_with_agent_state_dict():
    torch.manual_seed(0)
    envs = _envs()
    obs_dim = envs.single_observation_space.shape[0]

    trained = ContinuousActorCritic(envs, use_obs_norm=True)
    trained.update_norm(torch.randn(128, obs_dim) * 9.0 - 3.0)

    restored = ContinuousActorCritic(envs, use_obs_norm=True)
    restored.load_state_dict(trained.state_dict())

    x = torch.randn(4, obs_dim)
    torch.testing.assert_close(restored.normalize_obs(x), trained.normalize_obs(x))
    envs.close()


def test_eval_without_update_norm_keeps_stats_frozen():
    torch.manual_seed(0)
    envs = _envs()
    obs_dim = envs.single_observation_space.shape[0]
    agent = ContinuousActorCritic(envs, use_obs_norm=True)
    agent.update_norm(torch.randn(64, obs_dim))

    before = {k: v.clone() for k, v in agent.obs_normalizer.state_dict().items()}
    with torch.no_grad():
        agent.get_action_and_value(torch.randn(32, obs_dim) * 100.0, deterministic=True)
        agent.get_value(torch.randn(32, obs_dim) * 100.0)
    after = agent.obs_normalizer.state_dict()

    for k in before:
        torch.testing.assert_close(after[k], before[k])
    envs.close()


def test_deterministic_action_is_the_normalized_mean():
    torch.manual_seed(0)
    envs = _envs()
    obs_dim = envs.single_observation_space.shape[0]
    agent = ContinuousActorCritic(envs, use_obs_norm=True)
    agent.update_norm(torch.randn(64, obs_dim) * 10.0)

    x = torch.randn(6, obs_dim) * 10.0
    action, _, _, _ = agent.get_action_and_value(x, deterministic=True)
    torch.testing.assert_close(action, agent.actor_mean(agent.normalize_obs(x)))
    envs.close()
