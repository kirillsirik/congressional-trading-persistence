"""Household concentration and influence diagnostics (INF-010, INF-011).

Thresholds for the material-influence rule are parameters, not constants:
the binding values await HS-07 signoff.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def pooled_mean(frame: pd.DataFrame, value_col: str) -> float:
    return float(frame[value_col].mean())


def equal_household_estimate(
    frame: pd.DataFrame, household_col: str, value_col: str
) -> dict:
    """Household means first, then an unweighted mean across households."""
    means = frame.groupby(household_col)[value_col].mean()
    g = len(means)
    se = float(means.std(ddof=1) / np.sqrt(g)) if g > 1 else float("nan")
    return {
        "estimate": float(means.mean()),
        "se_across_households": se,
        "n_households": g,
    }


@dataclass
class LeaveOneOutRow:
    household: str
    estimate: float
    abs_change_fraction: float
    sign_reversed: bool
    exceeds_change_threshold: bool


def leave_one_household_out(
    frame: pd.DataFrame,
    household_col: str,
    value_col: str,
    households: list,
    change_threshold: float,
    estimator: Callable[[pd.DataFrame], float] | None = None,
) -> tuple[float, list[LeaveOneOutRow]]:
    """Re-estimate after deleting each pre-identified household.

    ``change_threshold`` is the HS-07 candidate (e.g. 0.50); it must be
    supplied by the caller because the binding value is unsigned.
    """
    est = estimator or (lambda df: pooled_mean(df, value_col))
    full = est(frame)
    rows = []
    for hh in households:
        sub = frame[frame[household_col] != hh]
        e = est(sub) if len(sub) else float("nan")
        change = abs(e - full) / abs(full) if full != 0 else float("inf")
        rows.append(
            LeaveOneOutRow(
                household=hh,
                estimate=e,
                abs_change_fraction=change,
                sign_reversed=bool(np.sign(e) != np.sign(full) and e != 0 and full != 0),
                exceeds_change_threshold=bool(change > change_threshold),
            )
        )
    return full, rows


def cluster_jackknife(
    frame: pd.DataFrame,
    cluster_col: str,
    estimator: Callable[[pd.DataFrame], float],
) -> dict:
    """Delete-one-cluster (CV3-style) estimates and variance."""
    clusters = pd.unique(frame[cluster_col])
    g = len(clusters)
    if g < 2:
        raise ValueError("cluster jackknife needs at least two clusters")
    estimates = np.array(
        [estimator(frame[frame[cluster_col] != c]) for c in clusters], dtype=float
    )
    center = estimates.mean()
    var = (g - 1) / g * float(((estimates - center) ** 2).sum())
    return {
        "estimates": pd.Series(estimates, index=pd.Index(clusters, name=cluster_col)),
        "variance": var,
        "se": float(np.sqrt(var)),
        "n_clusters": g,
    }


def household_concentration(
    frame: pd.DataFrame, household_col: str, weight_col: str | None = None
) -> dict:
    """Herfindahl, effective count, and top-1/5/10 activity shares."""
    if weight_col is None:
        activity = frame.groupby(household_col).size().astype(float)
    else:
        activity = frame.groupby(household_col)[weight_col].sum().astype(float)
    shares = (activity / activity.sum()).sort_values(ascending=False)
    hhi = float((shares**2).sum())
    return {
        "n_households": int(len(shares)),
        "herfindahl": hhi,
        "effective_household_count": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "top_1_share": float(shares.iloc[:1].sum()),
        "top_5_share": float(shares.iloc[:5].sum()),
        "top_10_share": float(shares.iloc[:10].sum()),
        "shares": shares,
    }
