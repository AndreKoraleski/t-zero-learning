"""Shared infrastructure used by train.py, evaluate.py, and the algorithms.

Modules:
- ``config_loader``: YAML config loading, CLI overrides, algorithm registry
- ``base_config``: config dataclasses shared by every algorithm
- ``checkpoint``: checkpoint save/load/resume and RNG state handling
- ``run_naming``: stable hashing for run directory names
"""
