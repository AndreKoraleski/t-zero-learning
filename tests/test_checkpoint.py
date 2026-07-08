"""Checkpoint envelope: save/load round-trip, rotation, RNG restore.

These cover the generic layer only — algorithm payloads are exercised by the
train/resume smoke tests.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from core.checkpoint import (
    apply_rng_state,
    delete_checkpoints,
    gather_rng_state,
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


def test_save_load_roundtrip(tmp_path):
    payload = {"weights": torch.tensor([1.0, 2.0]), "counter": 7}
    save_checkpoint(tmp_path, global_step=128, algorithm_state=payload)

    bundle = load_checkpoint(tmp_path / "checkpoint_gs128.pt")

    assert bundle["format_version"] == 2
    assert bundle["global_step"] == 128
    assert set(bundle["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    torch.testing.assert_close(bundle["algorithm"]["weights"], payload["weights"])
    assert bundle["algorithm"]["counter"] == 7


def test_rotation_keeps_newest(tmp_path):
    for step in (64, 128, 192, 256):
        save_checkpoint(tmp_path, step, {"s": step}, checkpoints_to_keep=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_gs*.pt"))
    assert remaining == ["checkpoint_gs192.pt", "checkpoint_gs256.pt"]
    assert latest_checkpoint_path(tmp_path).name == "checkpoint_gs256.pt"


def test_latest_checkpoint_sorts_numerically(tmp_path):
    """gs1000 > gs999 — must sort by step, not by string."""
    save_checkpoint(tmp_path, 999, {}, checkpoints_to_keep=10)
    save_checkpoint(tmp_path, 1000, {}, checkpoints_to_keep=10)
    assert latest_checkpoint_path(tmp_path).name == "checkpoint_gs1000.pt"


def test_delete_checkpoints(tmp_path):
    save_checkpoint(tmp_path, 64, {})
    delete_checkpoints(tmp_path)
    assert latest_checkpoint_path(tmp_path) is None


def test_unknown_format_version_raises(tmp_path):
    path = tmp_path / "checkpoint_gs1.pt"
    torch.save({"format_version": 99, "global_step": 1}, path)
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


def test_rng_restore_reproduces_streams():
    """Restoring RNG state must make all three generators replay identically —
    this is what makes a resumed run continue the same random sequence."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    state = gather_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    # advance the streams past the capture point
    random.random(), np.random.rand(), torch.rand(3)

    apply_rng_state(state, cuda=False)
    replayed = (random.random(), np.random.rand(), torch.rand(3))

    assert replayed[0] == expected[0]
    assert replayed[1] == expected[1]
    torch.testing.assert_close(replayed[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rng_restore_cuda():
    """Pins the fixed bug: set_rng_state_all needs CPU ByteTensors."""
    torch.cuda.manual_seed_all(42)
    state = gather_rng_state()
    expected = torch.rand(3, device="cuda")
    torch.rand(3, device="cuda")  # advance

    apply_rng_state(state, cuda=True)
    torch.testing.assert_close(torch.rand(3, device="cuda"), expected)
