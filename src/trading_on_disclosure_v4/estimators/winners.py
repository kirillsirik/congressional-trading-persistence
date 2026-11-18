"""A3 false-winner statistics on one panel and winner-identity turnover across draws (plan §7.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .tau import finite_outcome_rows

MIN_EVENTS_FULL_SAMPLE = 20


def household_stats(
    panel: pd.DataFrame, min_n: int = MIN_EVENTS_FULL_SAMPLE, y: str = "y_h20", household: str = "household_id"
) -> pd.DataFrame:
    """Per-household n, mean, sd, se, t and two-sided nominal p (t with n-1 dof); rows with n >= min_n.

    Non-finite outcomes are dropped first; the count is in ``frame.attrs['n_dropped_nonfinite']``."""
    panel, n_dropped = finite_outcome_rows(panel, y)
    g = panel.groupby(household, sort=True)[y].agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1))
    g = g.loc[g["n"] >= max(min_n, 2)].copy()
    g["se"] = g["sd"] / np.sqrt(g["n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        g["t"] = np.where(g["se"] > 0, g["mean"] / g["se"], np.nan)
    g["p"] = 2 * stats.t.sf(np.abs(g["t"]), df=g["n"] - 1)
    out = g.reset_index()
    out.attrs["n_dropped_nonfinite"] = n_dropped
    out.attrs["household"] = household
    return out


def winner_stats(hh: pd.DataFrame, score: str = "mean", alpha: float = 0.05, household: str | None = None) -> dict:
    """Largest mean, largest t, top-decile mean, count of p < alpha, winner identity by mean and by t.

    ``household`` names the id column (default: the one ``household_stats`` recorded, else 'household_id')."""
    if hh.empty:
        raise ValueError("no household meets the minimum event count")
    household = household or hh.attrs.get("household", "household_id")
    k = max(1, int(np.ceil(0.1 * len(hh))))
    top = hh.nlargest(k, score)
    i_mean, i_t = int(hh[score].idxmax()), int(hh["t"].idxmax()) if hh["t"].notna().any() else -1
    return {
        "n_households": int(len(hh)),
        "max_mean": float(hh[score].max()),
        "max_t": float(hh["t"].max()),
        "top_decile_mean": float(top[score].mean()),
        "top_decile_size": k,
        "n_nominal_significant": int((hh["p"] < alpha).sum()),
        "share_nominal_significant": float((hh["p"] < alpha).mean()),
        "winner_by_mean": str(hh.loc[i_mean, household]),
        "winner_by_t": str(hh.loc[i_t, household]) if i_t >= 0 else None,
        "score": score,
        "n_dropped_nonfinite": int(hh.attrs.get("n_dropped_nonfinite", 0)),
    }


def panel_winner_stats(
    panel: pd.DataFrame, min_n: int = MIN_EVENTS_FULL_SAMPLE, y: str = "y_h20", alpha: float = 0.05, household: str = "household_id"
) -> dict:
    return winner_stats(household_stats(panel, min_n=min_n, y=y, household=household), alpha=alpha, household=household)


def turnover(identities: list[str] | np.ndarray) -> dict:
    """Identity churn across draws: share of consecutive changes, distinct winners, top winner's share."""
    ids = np.asarray([str(i) for i in identities])
    if len(ids) == 0:
        raise ValueError("no identities")
    vals, counts = np.unique(ids, return_counts=True)
    changes = float(np.mean(ids[1:] != ids[:-1])) if len(ids) > 1 else 0.0
    return {
        "n_draws": int(len(ids)),
        "n_distinct_winners": int(len(vals)),
        "consecutive_change_share": changes,
        "top_winner_share": float(counts.max() / len(ids)),
        "top_winner": str(vals[counts.argmax()]),
    }
