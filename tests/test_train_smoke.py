"""End-to-end training smoke test: does the whole pipeline still work?

Parametrized over the ``ALGORITHMS`` registry — registering a new algorithm
automatically adds it here (and the guard test below fails until it has
smoke settings).  Asserts only the *mechanics* every algorithm shares: run
dir layout, config persistence, checkpoint envelope, finite state.  Never
asserts learning performance (RL is stochastic).
"""
from __future__ import annotations

import pytest
import torch

from core.checkpoint import latest_checkpoint_path, load_checkpoint
from core.config_loader import ALGORITHMS
from tests.helpers import SMOKE_SETTINGS, make_smoke_args, only_run_dir

pytestmark = pytest.mark.slow


def test_every_algorithm_has_smoke_settings():
    """Adding an algorithm to the registry without smoke coverage is an error."""
    missing = set(ALGORITHMS) - set(SMOKE_SETTINGS)
    assert not missing, (
        f"Algorithms without smoke settings: {sorted(missing)} — "
        "add an entry to SMOKE_SETTINGS in tests/helpers.py"
    )


def _assert_all_tensors_finite(obj, path="algorithm"):
    """Recursively check every tensor in a checkpoint payload (works for any
    algorithm's state layout — model weights, optimizer moments, buffers)."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            assert torch.isfinite(obj).all(), f"non-finite values in {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_tensors_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_all_tensors_finite(v, f"{path}[{i}]")


@pytest.mark.parametrize("algo_name", sorted(ALGORITHMS))
def test_train_smoke(algo_name, tmp_run_dir):
    args, main = make_smoke_args(algo_name)

    main(args)

    run_dir = only_run_dir(tmp_run_dir)

    # Run directory naming: env__kwargs-hash__seed__timestamp
    env_part = args.env_id.replace("/", "-")
    assert run_dir.name.startswith(f"{env_part}__empty__{args.seed}__")

    # Config persisted for resume / provenance
    assert (run_dir / "config.yml").is_file()

    # Checkpoints written and rotated
    ckpt_path = latest_checkpoint_path(run_dir)
    assert ckpt_path is not None
    ckpts = list(run_dir.glob("checkpoint_gs*.pt"))
    assert len(ckpts) <= args.checkpoints_to_keep

    # Checkpoint envelope + finite state (a nan anywhere means the training
    # loop diverged or a buffer was mis-wired)
    bundle = load_checkpoint(ckpt_path, torch.device("cpu"))
    assert bundle["format_version"] == 2
    assert bundle["global_step"] == args.total_timesteps
    _assert_all_tensors_finite(bundle["algorithm"])
