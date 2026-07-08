"""Run-name hashing stability.

``string_to_id`` output is baked into every run directory name on disk;
changing it would orphan existing runs, so the expected values are
hardcoded rather than recomputed.
"""
from core.run_naming import string_to_id


def test_known_value_is_stable():
    # sha256('{"target_vel": 0.5}')[:16] — do NOT update this without
    # accepting that existing run-dir names stop matching their kwargs.
    assert string_to_id('{"target_vel": 0.5}') == "78a22911b916ab5e"


def test_empty_string_maps_to_empty():
    assert string_to_id("") == "empty"


def test_fixed_length_and_hex():
    out = string_to_id('{"directions": [[1.0, 0.0], [-1.0, 0.0]]}')
    assert len(out) == 16
    int(out, 16)  # raises if not hex


def test_distinct_inputs_distinct_ids():
    assert string_to_id('{"a": 1}') != string_to_id('{"a": 2}')
