"""Config save → load round-trip.

The algorithm's dataclass field name equals the YAML section name
(``ppo_continuous_action:``), so a run's saved ``config.yml`` (written from
``asdict``) and hand-written configs are the same dialect — these tests keep
the round-trip lossless. Unknown keys in a config are hard errors.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest
import yaml

from algorithms.ppo_continuous_action import Args
from core.checkpoint import write_run_config_yaml
from core.config_loader import ALGORITHMS, _import_algorithm, load_config


@pytest.mark.parametrize("algo_name", sorted(ALGORITHMS))
def test_default_roundtrip_every_algorithm(algo_name, tmp_path):
    """Registry-driven: every algorithm's default Args must survive the
    save -> load cycle (covers new algorithms automatically)."""
    ArgsClass, _ = _import_algorithm(algo_name)
    args = ArgsClass()
    args.algorithm = algo_name
    write_run_config_yaml(str(tmp_path), "run", args)

    loaded, loaded_algo = load_config(str(tmp_path / "run" / "config.yml"))

    assert loaded_algo == algo_name
    assert asdict(loaded) == asdict(args)


@pytest.fixture
def saved_config(tmp_path):
    """Write a non-default Args the way a real run does, return its config.yml path.

    Uses the canonical PPO as the representative algorithm for exercising
    *non-default* nested values; the parametrized test above covers every
    registered algorithm with defaults.
    """
    args = Args()
    args.algorithm = "ppo_continuous_action"
    args.env_id = "HalfCheetahVel-v1"
    args.env_kwargs = {"target_vel": 0.5}
    args.seed = 777
    args.total_timesteps = 12345
    args.agent.activation = "ReLU"
    args.agent.hidden_layers_size = 128
    args.ppo_continuous_action.gamma = 0.98
    args.ppo_continuous_action.num_steps = 64
    write_run_config_yaml(str(tmp_path), "run", args)
    return args, tmp_path / "run" / "config.yml"


def test_roundtrip_field_equality(saved_config):
    original, config_path = saved_config
    loaded, algo_name = load_config(str(config_path))

    assert algo_name == "ppo_continuous_action"
    assert asdict(loaded) == asdict(original)


def test_roundtrip_resolves_nested_dataclasses(saved_config):
    """The nested sections must become dataclasses, not dicts."""
    _, config_path = saved_config
    loaded, _ = load_config(str(config_path))

    assert loaded.agent.activation == "ReLU"
    assert loaded.ppo_continuous_action.gamma == 0.98
    assert loaded.algo is loaded.ppo_continuous_action


def test_stale_keys_are_fatal(saved_config, capsys):
    """Unknown keys (e.g. fields removed since the run was saved) are hard
    errors: resuming such a run requires hand-editing its config.yml, which
    is deliberate — a typo must never silently fall back to defaults."""
    _, config_path = saved_config
    cfg = yaml.safe_load(config_path.read_text())
    cfg["ppo_continuous_action"]["separate_actor_critic_optim"] = False  # removed field
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(SystemExit):
        load_config(str(config_path))
    out = capsys.readouterr().out
    assert "Error" in out
    assert "Valid keys" in out


def test_unknown_top_level_key_is_fatal(saved_config, capsys):
    _, config_path = saved_config
    cfg = yaml.safe_load(config_path.read_text())
    cfg["some_removed_top_level_key"] = 42
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(SystemExit):
        load_config(str(config_path))
    assert "Error" in capsys.readouterr().out
