"""Running observation normalization owned by the agent (not the env).

Rationale: normalization statistics are part of what the policy *is* — a
checkpoint is only reproducible if the stats travel with the weights.  Keeping
them as registered buffers on an ``nn.Module`` means ``agent.state_dict()``
carries them automatically through every existing save path
(``model.pt`` and ``checkpoint_gs*.pt``) with no envelope changes.

This replaces ``gym.wrappers.NormalizeObservation`` (whose stats live in the
env and are silently lost on save/resume/eval).  Reward normalization stays
env-side (``envs.wrappers``): it only shapes the training signal and is never
needed at evaluation time.

Usage contract (see docs/adding-a-new-algorithm.md): the *algorithm* drives
the statistics — call ``agent.update_norm(obs)`` once per collected obs during
rollout, act on ``agent.normalize_obs(obs)``, and store the normalized obs.
Evaluation never calls ``update_norm``, so stats stay frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Tracks running mean, variance and count for vector observations.

    Uses Chan et al.'s parallel-variance update (same math as
    ``gym.wrappers.NormalizeObservation``).  State lives in registered
    buffers so it is saved/loaded via ``state_dict`` and moves with
    ``.to(device)``.
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)


class ObsNormalizer(nn.Module):
    """Running normalizer for flat observation tensors, with output clipping.

    ``normalize`` returns ``clip((x - mean) / sqrt(var + eps), ±clip)`` — the
    clip applies to the *normalized* values, matching CleanRL's wrapper order
    (``NormalizeObservation`` then ``TransformObservation(clip)``).
    """

    def __init__(self, obs_dim: int, epsilon: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.running = RunningMeanStd(shape=(self.obs_dim,))

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_flat = x.reshape(x.shape[0], -1)
        if x_flat.shape[1] != self.obs_dim:
            raise ValueError(
                f"ObsNormalizer expected obs_dim={self.obs_dim}, got {x_flat.shape[1]}"
            )
        return x_flat

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        self.running.update(self._flatten(x.detach()).to(torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self._flatten(x)
        mean = self.running.mean.to(device=x_flat.device, dtype=x_flat.dtype)
        var = self.running.var.to(device=x_flat.device, dtype=x_flat.dtype)
        x_norm = (x_flat - mean) / torch.sqrt(var + self.epsilon)
        x_norm = torch.clamp(x_norm, -self.clip, self.clip)
        return x_norm.reshape_as(x)
