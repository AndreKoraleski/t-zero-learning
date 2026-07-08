"""Environment construction: single-env thunks and vectorised training envs.

Public API
----------
- ``make_env``       — returns a zero-argument thunk that builds one wrapped env.
- ``build_vector_envs`` — creates an ``AsyncVectorEnv`` (or an env's own vectoriser).

Environment-specific quirks (custom vectorisers, wrapper skips, etc.) are
declared via :mod:`envs.adapters`; this module stays env-agnostic.
"""

from __future__ import annotations

import gymnasium as gym

import envs.custom_envs  # noqa: F401 — Gym registration side effects

from envs.adapters import get_adapter
from envs.video import wrap_eval_video, wrap_train_video


# ---------------------------------------------------------------------------
# Single-env thunk
# ---------------------------------------------------------------------------


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    gamma,
    experiment_dir=None,
    env_kwargs=None,
    video_every_global_steps: int = 0,
    num_envs: int = 1,
    video_length: int = 0,
    name_prefix="rl-video",
    wrappers=None,
):
    """Return a zero-argument thunk that creates a single wrapped environment.

    Only worker ``idx == 0`` records video.  The preprocessing stack applied
    to every worker is the env's adapter override (``EnvAdapter.apply_wrappers``)
    if registered, otherwise *wrappers* (the algorithm's ``default_wrappers``).
    Stacks are called as ``wrap(env, env_id, gamma)`` — *gamma* is the
    algorithm's discount factor, used by reward normalization.
    There is deliberately no implicit default: callers must say which stack
    they want (pass ``lambda env, env_id, gamma: env`` for a raw env).
    """

    def thunk():
        render_for_video = capture_video and idx == 0

        if render_for_video:
            env = gym.make(env_id, render_mode="rgb_array", **(env_kwargs or {}))
        else:
            env = gym.make(env_id, **(env_kwargs or {}))

        # --- video recording (worker 0 only) ---
        if render_for_video:
            video_dir = f"{experiment_dir}/{run_name}/videos/"
            if name_prefix == "eval":
                env = wrap_eval_video(env, video_dir)
            elif video_every_global_steps and video_every_global_steps > 0:
                env = wrap_train_video(
                    env,
                    video_dir,
                    num_envs=int(num_envs),
                    video_every_global_steps=int(video_every_global_steps),
                    video_length=int(video_length),
                    name_prefix=name_prefix,
                )

        wrap = get_adapter(env_id).apply_wrappers or wrappers
        if wrap is None:
            raise ValueError(
                f"{env_id}: no preprocessing wrapper stack specified. Pass "
                "wrappers= explicitly (e.g. envs.wrappers.continuous_control_"
                "wrappers, or `lambda env, env_id, gamma: env` for a raw env), "
                "or register an EnvAdapter with apply_wrappers for this env."
            )
        env = wrap(env, env_id, gamma)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Vectorised env builder
# ---------------------------------------------------------------------------


def build_vector_envs(
    env_id: str,
    env_kwargs: dict | None,
    seed: int,
    num_envs: int,
    capture_video: bool,
    run_name: str,
    gamma: float,
    experiment_dir: str | None,
    video_every_global_steps: int,
    video_length_steps: int,
    wrappers=None,
) -> tuple[gym.vector.VectorEnv, int]:
    """Build a vectorised training environment.

    If the env registers a custom vectoriser (see :mod:`envs.adapters`), it is
    used and the effective env count comes from the returned ``VectorEnv``.
    Otherwise an ``AsyncVectorEnv`` is built from per-worker ``make_env`` thunks,
    each applying the preprocessing stack (*wrappers* is the algorithm's
    default; per-env adapter overrides win — see :func:`make_env`).

    Returns ``(vector_env, effective_num_envs)``.
    """
    adapter = get_adapter(env_id)
    if adapter.make_vector_env is not None:
        envs = adapter.make_vector_env(env_id, env_kwargs or {}, seed)
        return envs, int(envs.num_envs)

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id,
                i,
                capture_video,
                run_name,
                gamma,
                experiment_dir,
                env_kwargs,
                video_every_global_steps=int(video_every_global_steps),
                num_envs=int(num_envs),
                video_length=int(video_length_steps),
                name_prefix="train",
                wrappers=wrappers,
            )
            for i in range(int(num_envs))
        ]
    )
    return envs, int(num_envs)
