"""Resume smoke test — the most fragile path in the repo.

Parametrized over the ``ALGORITHMS`` registry: trains a short run, then
resumes it from the saved config + checkpoint (the same flow
``train.py --resume`` uses) and asserts training actually continued.

The cross-variant guard test is intentionally specific to the two PPO
variants — it pins their friendly refuse-to-resume error.
"""
from __future__ import annotations

import pytest

from core.checkpoint import latest_checkpoint_path
from core.config_loader import ALGORITHMS, load_config
from tests.helpers import make_smoke_args, only_run_dir

pytestmark = pytest.mark.slow


def _checkpoint_step(run_dir) -> int:
    path = latest_checkpoint_path(run_dir)
    assert path is not None
    return int(path.stem.removeprefix("checkpoint_gs"))


@pytest.mark.parametrize("algo_name", sorted(ALGORITHMS))
def test_resume_continues_training(algo_name, tmp_run_dir):
    # Phase 1: fresh run to the smoke budget
    args, main = make_smoke_args(algo_name)
    budget = args.total_timesteps
    main(args)
    run_dir = only_run_dir(tmp_run_dir)
    assert _checkpoint_step(run_dir) == budget

    # Phase 2: resume from the saved run config (as train.py --resume does)
    # with a doubled budget
    resumed_args, resumed_algo = load_config(str(run_dir / "config.yml"))
    assert resumed_algo == algo_name
    resumed_args.total_timesteps = 2 * budget
    _, main_again = make_smoke_args(algo_name)  # fresh main fn reference
    main_again(resumed_args, resume_run_dir=run_dir)

    assert _checkpoint_step(run_dir) == 2 * budget
    # Still a single run directory — resume must not fork a new one
    assert only_run_dir(tmp_run_dir) == run_dir


def test_cross_variant_resume_raises(tmp_run_dir):
    """PPO-pair specific: resuming a single-optimizer checkpoint with the
    split-optimizer variant must fail with the friendly ValueError, not a
    cryptic KeyError."""
    args, main = make_smoke_args("ppo_continuous_action")
    main(args)
    run_dir = only_run_dir(tmp_run_dir)

    split_args, split_main = make_smoke_args(
        "ppo_continuous_action_split_optim", total_timesteps=512
    )
    with pytest.raises(ValueError, match="different algorithm"):
        split_main(split_args, resume_run_dir=run_dir)
