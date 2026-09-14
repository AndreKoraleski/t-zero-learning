"""YAML config loading, algorithm registry, and CLI override helpers.

Used by ``train.py`` — keeps the entry-point file thin.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Algorithm registry
#   algorithm_name -> (module_path, class_name, args_class_name)
#
#   The algorithm's hyperparameters live in a YAML section named exactly
#   after the algorithm (== the Args dataclass field name), so configs,
#   saved run configs, and CLI override paths all share one spelling.
# ---------------------------------------------------------------------------
ALGORITHMS: dict[str, tuple[str, str, str]] = {
    "ppo_continuous_action": ("algorithms.ppo_continuous_action", "PPO", "Args"),
    "ppo_continuous_action_split_optim": ("algorithms.ppo_continuous_action_split_optim", "PPO", "Args"),
    "dqn": ("algorithms.dqn", "DQN", "Args"),
}


def _import_algorithm(algo_name: str):
    """Return ``(ArgsClass, main_fn)`` for *algo_name*."""
    if algo_name not in ALGORITHMS:
        print(f"Error: unknown algorithm '{algo_name}'")
        print(f"Available algorithms: {', '.join(ALGORITHMS)}")
        sys.exit(1)
    module_path, _cls_name, args_cls_name = ALGORITHMS[algo_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, args_cls_name), getattr(mod, "main")


def _fail_unknown_key(what: str, target: Any) -> None:
    """Exit with the list of valid keys — a typo'd key must never silently
    start a run with default hyperparameters."""
    print(f"Error: {what}")
    if is_dataclass(target):
        print(f"Valid keys: {', '.join(f.name for f in fields(target))}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> tuple[Any, str]:
    """Load a YAML config and return ``(args, algo_name)``.

    The config must declare an ``algorithm`` key selecting the entry in
    :data:`ALGORITHMS`.  Every top-level key maps onto an ``Args`` field;
    nested dataclass fields (``agent:`` and the algorithm's own section,
    e.g. ``ppo_continuous_action:``) are set field-by-field.  Unknown keys
    are hard errors — a typo must fail at startup, not after a 50M-step run.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict) or "algorithm" not in config_dict:
        print(f"Error: config '{config_path}' must declare an 'algorithm' key")
        print(f"Available algorithms: {', '.join(ALGORITHMS)}")
        sys.exit(1)

    algo_name = config_dict["algorithm"]
    ArgsClass, _ = _import_algorithm(algo_name)
    args = ArgsClass()

    top_level = {f.name for f in fields(args)}
    for key, value in config_dict.items():
        if key not in top_level:
            _fail_unknown_key(f"unknown config key '{key}' in {config_path}", args)
        current = getattr(args, key)
        if is_dataclass(current) and isinstance(value, dict):
            sub_level = {f.name for f in fields(current)}
            for sub_key, sub_value in value.items():
                if sub_key not in sub_level:
                    _fail_unknown_key(
                        f"unknown config key '{key}.{sub_key}' in {config_path}", current
                    )
                setattr(current, sub_key, sub_value)
        else:
            setattr(args, key, value)

    return args, algo_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_config_path(config_name: str) -> str:
    """Resolve a config *name* to a file path in ``configs/``."""
    stem = Path(config_name).stem
    path = Path("configs") / f"{stem}.yml"
    if not path.exists():
        path = Path("configs") / f"{stem}.yaml"
    return str(path)


def get_experiment_name(config_name: str) -> str:
    return Path(config_name).stem


def apply_overrides(args: Any, overrides: list[str] | None) -> None:
    """Apply CLI overrides.  Supports dotted keys (``agent.activation=ReLU``).

    The algorithm section uses its field name (== the YAML section name,
    e.g. ``ppo_continuous_action.num_steps=512``); ``algo.`` works as an
    algorithm-agnostic shorthand (``algo.num_steps=512``).  Values for
    dict-typed fields (e.g. ``env_kwargs``) are parsed as JSON:
    ``--override 'env_kwargs={"target_vel": 0.5}'``.

    Malformed or unknown overrides are hard errors — never silently ignored.
    """
    if not overrides:
        return
    for override in overrides:
        if "=" not in override:
            print(f"Error: invalid override '{override}' (expected key=value)")
            sys.exit(1)

        key, value = override.split("=", 1)

        parts = key.split(".")
        target = args
        for part in parts[:-1]:
            if not hasattr(target, part):
                _fail_unknown_key(f"unknown config path '{key}' in override", target)
            target = getattr(target, part)
        attr = parts[-1]

        if not hasattr(target, attr):
            _fail_unknown_key(f"unknown config key '{key}' in override", target)

        current_value = getattr(target, attr)
        if isinstance(current_value, bool):
            setattr(target, attr, value.lower() in ("true", "1", "yes"))
        elif isinstance(current_value, int):
            setattr(target, attr, int(value))
        elif isinstance(current_value, float):
            setattr(target, attr, float(value))
        elif isinstance(current_value, dict):
            setattr(target, attr, json.loads(value))
        else:
            setattr(target, attr, value)
        print(f"Override: {key} = {value}")
