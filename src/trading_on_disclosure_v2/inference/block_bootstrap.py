"""Calendar-block bootstrap over entry dates (INF-006..008).

Resamples whole entry-date blocks and carries the full cross-section of
signals on each selected date, so adjacent-date overlapping-window dependence
survives into the resample. Seed streams derive deterministically from the
freeze hash (CFG-005); resample index matrices are persistable and replayable
and can be shared across a family for joint inference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIMARY_BLOCK_LENGTH = 20
SENSITIVITY_BLOCK_LENGTHS = (10, 40)
DEFAULT_FINAL_DRAWS = 4999


def seed_from_freeze_hash(freeze_hash: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{freeze_hash}|{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def draw_block_indices(
    n_dates: int,
    n_draws: int,
    block_length: int,
    freeze_hash: str,
    purpose: str,
    method: str = "moving_block",
) -> np.ndarray:
    """(n_draws, n_dates) matrix of date positions; identical for equal seeds."""
    if n_dates < 1:
        raise ValueError("need at least one entry date")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    rng = np.random.default_rng(seed_from_freeze_hash(freeze_hash, purpose))
    L = min(block_length, n_dates)
    out = np.empty((n_draws, n_dates), dtype=np.int64)
    if method == "moving_block":
        n_blocks = -(-n_dates // L)
        starts = rng.integers(0, n_dates - L + 1, size=(n_draws, n_blocks))
        for d in range(n_draws):
            seq = (starts[d][:, None] + np.arange(L)[None, :]).ravel()
            out[d] = seq[:n_dates]
    elif method == "stationary":
        # Politis-Romano: geometric block lengths (mean L), wrap-around.
        for d in range(n_draws):
            pos, filled = [], 0
            while filled < n_dates:
                start = int(rng.integers(0, n_dates))
                length = int(rng.geometric(1.0 / L))
                block = (start + np.arange(length)) % n_dates
                pos.append(block)
                filled += length
            out[d] = np.concatenate(pos)[:n_dates]
    else:
        raise ValueError(f"unknown bootstrap method: {method!r}")
    return out


@dataclass
class BlockBootstrapResult:
    statistics: np.ndarray
    indices: np.ndarray
    unique_dates: pd.DatetimeIndex
    block_length: int
    method: str
    seed_purpose: str

    def percentile_interval(self, level: float = 0.95) -> tuple[float, float]:
        a = (1.0 - level) / 2.0
        lo, hi = np.nanquantile(self.statistics, [a, 1.0 - a])
        return float(lo), float(hi)

    def percentile_t_interval(
        self, estimate: float, se: float, boot_se: np.ndarray, level: float = 0.95
    ) -> tuple[float, float]:
        t = (self.statistics - estimate) / np.where(boot_se > 0, boot_se, np.nan)
        a = (1.0 - level) / 2.0
        t_lo, t_hi = np.nanquantile(t, [a, 1.0 - a])
        return float(estimate - t_hi * se), float(estimate - t_lo * se)


def calendar_block_bootstrap(
    frame: pd.DataFrame,
    date_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    freeze_hash: str,
    purpose: str,
    block_length: int = PRIMARY_BLOCK_LENGTH,
    n_draws: int = DEFAULT_FINAL_DRAWS,
    method: str = "moving_block",
    indices: np.ndarray | None = None,
) -> BlockBootstrapResult:
    """Bootstrap ``stat_fn`` by resampling whole entry-date blocks.

    Pass a precomputed ``indices`` matrix to share draws across a family
    (INF-008); it overrides ``n_draws``/seed derivation.
    """
    dates = pd.DatetimeIndex(pd.unique(frame[date_col])).sort_values()
    if indices is None:
        indices = draw_block_indices(
            len(dates), n_draws, block_length, freeze_hash, purpose, method
        )
    rows_by_date = {d: sub for d, sub in frame.groupby(date_col, sort=False)}
    stats = np.empty(indices.shape[0])
    for i, draw in enumerate(indices):
        parts = [rows_by_date[dates[j]] for j in draw]
        stats[i] = stat_fn(pd.concat(parts, ignore_index=True))
    return BlockBootstrapResult(
        statistics=stats,
        indices=indices,
        unique_dates=dates,
        block_length=block_length,
        method=method,
        seed_purpose=purpose,
    )
