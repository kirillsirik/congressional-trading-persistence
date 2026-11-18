"""Filesystem IO with atomic writes and overwrite protection.

Dropbox local-sync mode treats the synced folder as an ordinary filesystem, so
these helpers are the single write path for both plain-local and Dropbox-local
setups. Raw snapshot folders are immutable: writes into an existing snapshot
directory are refused.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temp file in the target directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_dataframe(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool = False,
    index: bool = False,
) -> Path:
    """Atomically write a DataFrame as CSV or Parquet by extension.

    Refuses to clobber an existing file unless ``overwrite=True`` — callers
    that regenerate processed data must opt in explicitly, and raw snapshots
    should never pass ``overwrite=True``.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists. Pass overwrite=True (processed data) or write a "
            "new versioned/timestamped file (raw snapshots are immutable)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(tmp_path, index=index)
        elif path.suffix == ".csv":
            df.to_csv(tmp_path, index=index)
        else:
            raise ValueError(f"Unsupported extension: {path.suffix}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def read_dataframe(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported extension: {path.suffix}")
