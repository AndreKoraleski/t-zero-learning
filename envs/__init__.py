"""Training-time vector env helpers plus packaged Gym registrations.

- ``factory``: single-env thunks and vectorised env construction.
- ``wrappers``: standard wrapper stack (obs/reward transforms, episode stats).
- ``video``: video recording wrappers and schedule resolution.
- ``adapters``: per-env behaviour overrides (custom vectoriser, eval, etc.);
  imported here so bundled env adapters register on import.
- ``custom_envs``: Gym-registered environment implementations.
"""

import envs.adapters  # noqa: F401 — registers bundled env adapters on import
from envs.factory import (
    build_vector_envs,
    make_env,
)
from envs.video import resolve_training_video_schedule
from envs.wrappers import continuous_control_wrappers

__all__ = [
    "build_vector_envs",
    "continuous_control_wrappers",
    "make_env",
    "resolve_training_video_schedule",
]
