"""Immutable content-addressed raw archive.

Every official response body is stored once under its SHA-256; a later fetch
with different bytes is a new object, never an overwrite. Metadata lives in
the fetch manifest (db.py); this module only handles bytes on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..io.local_io import atomic_write_bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store_bytes(raw_store: Path, data: bytes, suffix: str = "") -> tuple[str, Path]:
    """Store bytes content-addressed; returns (sha256, path). Idempotent."""
    digest = sha256_bytes(data)
    path = raw_store / digest[:2] / f"{digest}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, data)
    return digest, path


def load_bytes(raw_store: Path, digest: str, suffix: str = "") -> bytes:
    path = raw_store / digest[:2] / f"{digest}{suffix}"
    return path.read_bytes()


def object_path(raw_store: Path, digest: str, suffix: str = "") -> Path:
    return raw_store / digest[:2] / f"{digest}{suffix}"
