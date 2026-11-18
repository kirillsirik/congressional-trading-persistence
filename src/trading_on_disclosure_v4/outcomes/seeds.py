"""Deterministic integer seeds derived from names (never from Python's hash)."""

from __future__ import annotations

import hashlib


def seed_from(*parts: object) -> int:
    """First 8 hex digits of sha256('|'.join(parts)) as an int."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
