"""Algorithm-agnostic training checkpoint I/O.

Checkpoint structure (``checkpoint_gs{step}.pt``)::

    {
        "format_version": 2,
        "global_step": int,
        "rng": { python, numpy, torch_cpu, torch_cuda },
        "algorithm": { ... algorithm-specific payload ... },
    }

Each algorithm class supplies its own slice via ``checkpoint_state_dict()``
and consumes it via ``load_checkpoint_state_dict(state)``.  This module only
handles the generic envelope: file I/O, RNG serialization, the
``format_version`` sanity check, and housekeeping (rotation, cleanup).
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

# Bumped only if the envelope structure above ever changes; the loader
# refuses checkpoints written with a different structure.
CHECKPOINT_FORMAT_VERSION = 2

_CHECKPOINT_RE = re.compile(r"^checkpoint_gs(\d+)\.pt$")


# ------------------------------------------------------------------
# RNG helpers
# ------------------------------------------------------------------

def gather_rng_state() -> dict[str, Any]:
    """Capture main-process RNG state (Python, NumPy, PyTorch CPU/CUDA)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.cpu() for t in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    return state


def apply_rng_state(rng_state: Mapping[str, Any], cuda: bool) -> None:
    """Restore main-process RNG state from a checkpoint ``rng`` dict."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if cuda and rng_state.get("torch_cuda") is not None:
        # set_rng_state_all expects CPU ByteTensors and copies to each device.
        torch.cuda.set_rng_state_all([t.cpu() for t in rng_state["torch_cuda"]])


# ------------------------------------------------------------------
# Config YAML
# ------------------------------------------------------------------

def write_run_config_yaml(experiment_dir: str, run_name: str, args: Any) -> None:
    """Persist the training configuration as ``config.yml`` inside the run directory."""
    from dataclasses import asdict, fields
    run_dir = Path(experiment_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = run_dir / "config.yml"
    # Use asdict for nested dataclasses; fall back to vars for non-dataclass objects
    try:
        cfg = asdict(args)
    except TypeError:
        cfg = vars(args)
    with open(config_save_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_save_path}")



# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

def _checkpoint_sort_key(path: Path) -> int:
    m = _CHECKPOINT_RE.match(path.name)
    return int(m.group(1)) if m else 0


def save_checkpoint(
    run_dir: Path,
    global_step: int,
    algorithm_state: dict[str, Any],
    checkpoints_to_keep: int = 3,
) -> Path:
    """Save a training checkpoint and rotate old files.

    Parameters
    ----------
    run_dir:
        Directory where ``checkpoint_gs{step}.pt`` files live.
    global_step:
        Current training step — used in the filename.
    algorithm_state:
        Algorithm-specific payload (from ``algorithm.checkpoint_state_dict()``).
    checkpoints_to_keep:
        Keep at most this many ``.pt`` files; oldest are deleted.

    Returns
    -------
    Path to the newly saved checkpoint file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "global_step": int(global_step),
        "rng": gather_rng_state(),
        "algorithm": algorithm_state,
    }
    path = run_dir / f"checkpoint_gs{global_step}.pt"
    torch.save(bundle, path)

    keep = max(1, int(checkpoints_to_keep))
    ckpts = sorted(run_dir.glob("checkpoint_gs*.pt"), key=_checkpoint_sort_key)
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint file.

    Returns a dict with keys ``format_version``, ``global_step``, ``rng``,
    and ``algorithm``.
    """
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict checkpoint at {path}, got {type(bundle)}")

    version = bundle.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version {version!r} in {path} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})"
        )
    return bundle


def latest_checkpoint_path(run_dir: Path) -> Path | None:
    """Return the newest ``checkpoint_gs*.pt`` in *run_dir*, or ``None``."""
    pt = sorted(run_dir.glob("checkpoint_gs*.pt"), key=_checkpoint_sort_key)
    return pt[-1] if pt else None


def delete_checkpoints(run_dir: Path) -> None:
    """Remove all ``checkpoint_gs*.pt`` files (called after final model save)."""
    for p in run_dir.glob("checkpoint_gs*.pt"):
        p.unlink(missing_ok=True)
