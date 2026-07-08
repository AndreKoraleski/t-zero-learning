"""Guard: docs/config-reference.md must match the config dataclasses.

The reference is generated from the dataclass field docstrings by
``scripts/generate_config_reference.py``.  Adding or changing a config field
without regenerating the doc fails here, with the fix in the message.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_config_reference_up_to_date():
    result = subprocess.run(
        [sys.executable, "scripts/generate_config_reference.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "docs/config-reference.md is stale — run: "
        "python scripts/generate_config_reference.py\n"
        f"{result.stdout}{result.stderr}"
    )
