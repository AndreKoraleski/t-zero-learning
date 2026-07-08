"""The adapter mechanism: the :class:`EnvAdapter` dataclass and the registry.

This module holds the machinery only; individual env adapters live in sibling
modules (e.g. ``envs/adapters/metaworld.py``) and are wired up in
``envs/adapters/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import gymnasium as gym

    from algorithms.base import Algorithm


@dataclass(frozen=True)
class EnvAdapter:
    """Declarative overrides for a family of environments.

    Every field has a default that means "standard Gymnasium behaviour", so an
    empty ``EnvAdapter()`` changes nothing.  Only set the fields your env needs.
    """

    make_vector_env: Callable[[str, dict, int], "gym.vector.VectorEnv"] | None = None
    """Custom vectoriser ``(env_id, env_kwargs, seed) -> VectorEnv``.

    Set this when the env ships its own vector entry point (e.g. multi-task
    benchmarks built with ``gym.make_vec``).  When ``None`` the framework builds
    an ``AsyncVectorEnv`` from per-worker ``make_env`` thunks.
    """

    apply_wrappers: Callable[["gym.Env", str, float], "gym.Env"] | None = None
    """Replacement preprocessing stack ``(env, env_id, gamma) -> env``.

    Set this when the env family needs different preprocessing than the
    algorithm's default stack (e.g. Atari frame preprocessing instead of
    flatten + clip, or no wrappers at all: ``lambda env, env_id, gamma: env``).
    It **replaces** the whole stack rather than composing with it; call
    :func:`envs.wrappers.continuous_control_wrappers` yourself for partial
    reuse.  When ``None`` the algorithm's ``default_wrappers`` applies.
    """

    skip_episode_stats: bool = False
    """If ``True``, do not add ``RecordEpisodeStatistics`` — the env already
    records episode returns/lengths itself (double-wrapping asserts on reset)."""

    supports_training_video: bool = True
    """If ``False``, disable per-step ``RecordVideo`` during training (e.g. envs
    built via ``gym.make_vec`` with no per-worker render surface)."""

    evaluate: Callable[["Algorithm", str, int, bool], dict] | None = None
    """Custom evaluation protocol.

    Signature ``(algo, model_path, eval_episodes, deterministic) -> dict`` where
    the dict has keys ``episodic_returns`` (list[float]) and ``metrics`` (dict
    for wandb).  When ``None`` the framework runs the standard single-env
    evaluation loop.
    """


# Default adapter shared by every env that registers nothing.
_DEFAULT_ADAPTER = EnvAdapter()

# Registered (prefix, adapter) pairs. Longest matching prefix wins.
_REGISTRY: list[tuple[str, EnvAdapter]] = []


def register_adapter(env_id_prefix: str, adapter: EnvAdapter) -> None:
    """Register *adapter* for every env-id starting with *env_id_prefix*.

    Re-registering the same prefix overwrites the previous adapter.
    """
    global _REGISTRY
    _REGISTRY = [(p, a) for (p, a) in _REGISTRY if p != env_id_prefix]
    _REGISTRY.append((env_id_prefix, adapter))


def get_adapter(env_id: str) -> EnvAdapter:
    """Return the adapter for *env_id* (longest matching prefix), or the default.

    Standard environments match nothing and receive :data:`_DEFAULT_ADAPTER`,
    whose fields all mean "behave like a plain Gymnasium env".
    """
    best_prefix = ""
    best_adapter = _DEFAULT_ADAPTER
    for prefix, adapter in _REGISTRY:
        if env_id.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_adapter = adapter
    return best_adapter
