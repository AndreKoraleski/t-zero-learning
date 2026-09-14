# Unit tests for the DQN assignment. Run with:  python -m pytest tests/test_dqn.py
# (Instructor: verify against the solution with
#  DQN_MODULE=algorithms.dqn_solution python -m pytest tests/test_dqn.py)
import importlib
import os

import numpy as np
import pytest
import torch

dqn = importlib.import_module(os.environ.get("DQN_MODULE", "algorithms.dqn"))

OBS_DIM = 3


def make_buffer(capacity=8):
    return dqn.ReplayBuffer(capacity, (OBS_DIM,), torch.device("cpu"))


def add_marked_transition(rb, i):
    """Transition whose fields are all derived from i, so samples can be validated."""
    obs = np.full(OBS_DIM, float(i), dtype=np.float32)
    rb.add(obs, obs + 0.5, i % 2, 10.0 * i, float(i % 3 == 0))


# ---------------------------------------------------------------- Part 1: buffer


def test_buffer_sample_shapes_and_dtypes():
    rb = make_buffer(capacity=8)
    for i in range(5):
        add_marked_transition(rb, i)
    batch = rb.sample(4)
    assert batch.observations.shape == (4, OBS_DIM)
    assert batch.next_observations.shape == (4, OBS_DIM)
    assert batch.actions.shape == (4, 1) and batch.actions.dtype == torch.int64
    assert batch.rewards.shape == (4, 1)
    assert batch.dones.shape == (4, 1)


def test_buffer_sampled_transitions_are_consistent():
    rb = make_buffer(capacity=16)
    for i in range(10):
        add_marked_transition(rb, i)
    batch = rb.sample(64)
    for k in range(64):
        i = int(batch.observations[k, 0].item())
        assert torch.allclose(batch.observations[k], torch.full((OBS_DIM,), float(i)))
        assert torch.allclose(batch.next_observations[k], batch.observations[k] + 0.5)
        assert batch.actions[k].item() == i % 2
        assert batch.rewards[k].item() == pytest.approx(10.0 * i)
        assert batch.dones[k].item() == pytest.approx(float(i % 3 == 0))


def test_buffer_never_samples_empty_slots():
    rb = make_buffer(capacity=8)
    for i in range(1, 4):  # only transitions 1, 2, 3 stored
        add_marked_transition(rb, i)
    batch = rb.sample(100)
    seen = {int(v) for v in batch.observations[:, 0].tolist()}
    assert seen <= {1, 2, 3}, f"sampled uninserted transitions: {seen - {1, 2, 3}}"


def test_buffer_fifo_overwrite_when_full():
    rb = make_buffer(capacity=4)
    for i in range(6):  # transitions 0 and 1 must have been overwritten
        add_marked_transition(rb, i)
    assert rb.size == 4
    batch = rb.sample(200)
    seen = {int(v) for v in batch.observations[:, 0].tolist()}
    assert seen <= {2, 3, 4, 5}, f"oldest transitions were not overwritten: {seen}"
    assert seen == {2, 3, 4, 5}, f"some surviving transitions are never sampled: {seen}"


# ------------------------------------------------------------- Part 2: TD target


def _stub_q(obs):
    """Deterministic 2-action 'target network': Q(s) = [s0, -s0] → max = |s0|."""
    return torch.stack([obs[:, 0], -obs[:, 0]], dim=1)


def make_batch(next_obs_vals, rewards, dones):
    n = len(rewards)
    next_obs = torch.zeros(n, OBS_DIM)
    next_obs[:, 0] = torch.tensor(next_obs_vals)
    return dqn.Batch(
        observations=torch.zeros(n, OBS_DIM),
        actions=torch.zeros(n, 1, dtype=torch.int64),
        next_observations=next_obs,
        rewards=torch.tensor(rewards).reshape(n, 1),
        dones=torch.tensor(dones).reshape(n, 1),
    )


def test_td_target_terminal_equals_reward():
    batch = make_batch(next_obs_vals=[10.0, 20.0, 30.0], rewards=[1.0, -2.0, 5.0], dones=[1.0, 1.0, 1.0])
    targets = dqn.compute_td_targets(_stub_q, batch, gamma=0.99)
    assert targets.shape == (3,)
    assert torch.allclose(targets, torch.tensor([1.0, -2.0, 5.0]))


def test_td_target_nonterminal_bootstraps_on_best_action():
    # max_a Q = |s0|: picking any fixed action instead of the max gives wrong values
    batch = make_batch(next_obs_vals=[2.0, -4.0], rewards=[1.0, 0.0], dones=[0.0, 0.0])
    targets = dqn.compute_td_targets(_stub_q, batch, gamma=0.5)
    assert torch.allclose(targets, torch.tensor([1.0 + 0.5 * 2.0, 0.5 * 4.0]))


def test_td_target_mixed_batch_shape_and_values():
    # shape must be (B,): unflattened (B,1) rewards/dones broadcast into (B,B)
    batch = make_batch(
        next_obs_vals=[10.0, 10.0, -10.0, -10.0],
        rewards=[1.0, 1.0, 1.0, 1.0],
        dones=[0.0, 1.0, 0.0, 1.0],
    )
    targets = dqn.compute_td_targets(_stub_q, batch, gamma=0.9)
    assert targets.shape == (4,)
    assert torch.allclose(targets, torch.tensor([10.0, 1.0, 10.0, 1.0]))
