"""Shared pytest fixtures.

Tests import repo modules directly (``config_loader``, ``checkpoint``, ...),
so the repo root is put on ``sys.path`` here — the same "run from repo root"
convention as ``train.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _no_experiment_tracking(monkeypatch):
    """Never let a test reach wandb servers, even if a config sets track=true."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def repo_root(monkeypatch) -> Path:
    """chdir to the repo root (``get_config_path`` and ``runs/`` are CWD-relative)."""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture
def tmp_run_dir(monkeypatch, tmp_path) -> Path:
    """chdir into a temp dir so training output (``runs/...``) never pollutes the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
