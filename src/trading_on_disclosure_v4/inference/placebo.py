"""I3: placebo p-value = position of the observed statistic in a vector of null (block-shift) values."""

from __future__ import annotations

import numpy as np


def placebo_p_value(observed: float, null_values: np.ndarray, alternative: str = "greater") -> dict:
    """Monte Carlo p with the add-one convention: (1 + #{null at least as extreme}) / (n + 1)."""
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    n = len(null)
    if n == 0:
        raise ValueError("null_values has no finite entries")
    if not np.isfinite(observed):
        raise ValueError("observed statistic is not finite")
    if alternative == "greater":
        extreme = int(np.sum(null >= observed))
    elif alternative == "less":
        extreme = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        center = float(np.median(null))
        extreme = int(np.sum(np.abs(null - center) >= abs(observed - center)))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    p = (1.0 + extreme) / (n + 1.0)
    return {
        "p": float(p), "n_null": n, "n_extreme": extreme, "alternative": alternative,
        "observed": float(observed), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if n > 1 else np.nan,
        "null_q95": float(np.quantile(null, 0.95)), "null_q99": float(np.quantile(null, 0.99)),
        "rank_from_top": extreme + 1,
    }


def null_critical_value(null_values: np.ndarray, alpha: float = 0.05) -> float:
    """Upper critical value at level alpha from the null distribution (used by T3 max-t)."""
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        raise ValueError("null_values has no finite entries")
    return float(np.quantile(null, 1.0 - alpha))
