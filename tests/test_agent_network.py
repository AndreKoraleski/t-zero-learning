"""ContinuousActorCritic: shapes, activation resolution, init determinism."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from networks import ContinuousActorCritic


def _envs(env_id="Pendulum-v1", n=2):
    return gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n)])


def test_output_shapes():
    envs = _envs()
    agent = ContinuousActorCritic(envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    x = torch.zeros(5, obs_dim)

    action, logprob, entropy, value = agent.get_action_and_value(x)

    assert action.shape == (5, act_dim)
    assert logprob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5, 1)
    assert agent.get_value(x).shape == (5, 1)
    envs.close()


def test_evaluating_given_action_returns_it():
    envs = _envs()
    agent = ContinuousActorCritic(envs)
    x = torch.zeros(4, envs.single_observation_space.shape[0])
    fixed = torch.ones(4, envs.single_action_space.shape[0])

    action, _, _, _ = agent.get_action_and_value(x, action=fixed)

    torch.testing.assert_close(action, fixed)
    envs.close()


@pytest.mark.parametrize("activation", ["Tanh", "ReLU", "GELU", "SiLU"])
def test_activation_string_resolves(activation):
    envs = _envs()
    agent = ContinuousActorCritic(envs, activation=activation)
    expected = getattr(torch.nn, activation)
    assert any(isinstance(m, expected) for m in agent.actor_mean.modules())
    envs.close()


def test_unknown_activation_fails():
    envs = _envs()
    with pytest.raises(AttributeError):
        ContinuousActorCritic(envs, activation="NotAnActivation")
    envs.close()


def test_hidden_layers_size():
    envs = _envs()
    agent = ContinuousActorCritic(envs, hidden_layers_size=128)
    hidden = [m for m in agent.critic.modules() if isinstance(m, torch.nn.Linear)]
    assert hidden[0].out_features == 128
    assert hidden[1].in_features == 128
    envs.close()


def test_seeded_init_is_deterministic():
    envs = _envs()
    torch.manual_seed(7)
    a = ContinuousActorCritic(envs)
    torch.manual_seed(7)
    b = ContinuousActorCritic(envs)

    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pa, pb, msg=f"param {name} differs across seeds")
    envs.close()


def test_actor_logstd_starts_at_zero():
    envs = _envs()
    agent = ContinuousActorCritic(envs)
    assert torch.all(agent.actor_logstd == 0.0)
    envs.close()
