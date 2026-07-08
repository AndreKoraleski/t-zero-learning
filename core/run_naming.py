"""Stable string hashing used to build run directory names."""

import hashlib


def string_to_id(text: str) -> str:
    """
    Map kwargs JSON (or any string) to a short fixed-length identifier for paths and run names.

    Uses SHA-256 truncated to hex so long env kwargs (e.g. large ``direction_pool``) do not
    blow past filesystem filename limits; the full kwargs remain in the saved config YAML.

    Args:
        text: Typically the canonical JSON encoding of ``env_kwargs``.

    Returns:
        16 hex characters (64-bit prefix of the hash), or ``empty`` for an empty string.
    """
    if not text:
        return "empty"
    return hashlib.sha256(text.encode()).hexdigest()[:16]
