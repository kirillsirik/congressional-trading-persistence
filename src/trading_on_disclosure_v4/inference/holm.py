"""Holm step-down adjustment of a family of p-values (missing p-values stay missing and do not count)."""

from __future__ import annotations

import numpy as np


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values: sort the m finite p's ascending, adj_(i) = max_{j <= i} (m - j + 1) p_(j), capped at 1.

    NaN entries are returned as NaN and do not enter m."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    scaled = (m - np.arange(m)) * p[order]
    out[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return out
