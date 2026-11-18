"""T1-T4 test statistics on one panel (plan §6): mean, variance component, max-t, upper-tail shares."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..inference import cgm
from ..inference.placebo import placebo_p_value
from . import winners
from .tau import PERIOD_A, PERIOD_B, finite_outcome_rows

Z_THRESHOLDS = (2.0, 3.0)


def lower_tail_stat(t: float) -> str:
    """Name of the share of standardized outcomes below -t (`share_lt_minus_3` beside `share_gt_3`)."""
    return f"share_lt_minus_{t:g}"


def t1_clustered_mean(
    panel: pd.DataFrame, y: str = "y_h20", household: str = "household_id", anchor: str = "anchor_date",
    weight: str | None = None, alpha: float = 0.05, periods: tuple[str, ...] = (PERIOD_A, PERIOD_B),
) -> dict:
    """T1: mean of y with two-way CGM SE (household x anchor date); one-sided (positive) and two-sided p.

    Non-finite outcomes are dropped first and counted (``n_dropped_nonfinite``)."""
    sub = panel.loc[panel["period"].isin(periods)] if "period" in panel.columns else panel
    sub, n_dropped = finite_outcome_rows(sub, y)
    yv = sub[y].to_numpy(dtype=float)
    w = sub[weight].to_numpy(dtype=float) if weight else np.ones(len(sub))
    clusters = pd.DataFrame({household: sub[household].to_numpy(), anchor: sub[anchor].to_numpy()})
    sl = cgm.clustered_slope(yv, np.ones((len(yv), 1)), w, clusters, coef_index=0)
    p_one = float(stats.t.sf(sl.t_stat, df=sl.dof)) if np.isfinite(sl.t_stat) else np.nan
    return {"test": "T1", "estimate": sl.coef, "se": sl.se, "t_stat": sl.t_stat, "p_two_sided": sl.p,
            "p_one_sided": p_one, "dof": sl.dof, "n_events": sl.n_rows, "n_households": sl.cluster_counts.get(household, 0),
            "n_anchor_dates": sl.cluster_counts.get(anchor, 0), "ci_low": sl.ci_low, "ci_high": sl.ci_high,
            "reject_two_sided": bool(sl.p < alpha), "reject_one_sided": bool(p_one < alpha), "n_dropped_nonfinite": n_dropped}


def t2_tau2_one_sided(panel: pd.DataFrame, weights: str = "ES", alpha: float = 0.05, **kw) -> dict:
    """T2: one-sided test of tau2 > 0 via I1 (t on b_hat with the two-way clustered SE)."""
    r = cgm.tau2_i1(panel, weights, **kw)
    r["test"] = "T2"
    r["reject_one_sided"] = bool(r["p_one_sided_tau2_positive"] < alpha)
    r["covers_zero_two_sided"] = bool(r["tau2_low"] <= 0.0 <= r["tau2_high"])
    return r


def max_t_statistic(panel: pd.DataFrame, min_n: int = winners.MIN_EVENTS_FULL_SAMPLE, y: str = "y_h20") -> dict:
    hh = winners.household_stats(panel, min_n=min_n, y=y)
    if hh.empty:
        return {"max_t": np.nan, "n_households": 0}
    i = int(hh["t"].idxmax())
    return {"max_t": float(hh.loc[i, "t"]), "n_households": int(len(hh)), "argmax_household": str(hh.loc[i, "household_id"])}


def t3_max_t(panel: pd.DataFrame, critical_value: float, min_n: int = winners.MIN_EVENTS_FULL_SAMPLE, y: str = "y_h20") -> dict:
    """T3: largest household t against a critical value taken from the null (N2) distribution of max t."""
    r = max_t_statistic(panel, min_n=min_n, y=y)
    r.update({"test": "T3", "critical_value": float(critical_value), "reject": bool(np.isfinite(r["max_t"]) and r["max_t"] > critical_value)})
    return r


def standardized_outcome(panel: pd.DataFrame, y: str = "y_h20", scale: str | None = "sigma_event") -> np.ndarray:
    """z = Y / (estimation-window residual SD x sqrt(20)); ``scale`` is that per-event denominator.

    Falls back to the pooled SD of Y when the column is absent (flagged by the caller as 'pooled').
    Rows with a non-finite outcome must already be removed (``t4_tail_shares`` does so)."""
    yv = panel[y].to_numpy(dtype=float)
    if scale and scale in panel.columns:
        s = panel[scale].to_numpy(dtype=float)
        return yv / s
    return yv / float(np.std(yv, ddof=1))


def t4_tail_shares(panel: pd.DataFrame, y: str = "y_h20", scale: str | None = "sigma_event", thresholds: tuple[float, ...] = Z_THRESHOLDS) -> dict:
    """T4 statistics: share of events with z above each threshold (raw Y, never winsorized), and beside it
    the share below minus the threshold (`lower_tail_stat`).

    Non-finite outcomes are dropped first and counted; a non-finite ``scale`` entry is a ValueError."""
    panel, n_dropped = finite_outcome_rows(panel, y)
    z = standardized_outcome(panel, y=y, scale=scale)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{int((~np.isfinite(z)).sum())} standardized outcomes are not finite (check {scale!r})")
    out = {"test": "T4", "n_events": int(len(z)), "scale": scale if (scale and scale in panel.columns) else "pooled_sd",
           "n_dropped_nonfinite": n_dropped}
    for t in thresholds:
        out[f"share_gt_{t:g}"] = float(np.mean(z > t))
        out[lower_tail_stat(t)] = float(np.mean(z < -t))
    return out


def t4_test(observed: dict, null_shares: pd.DataFrame, alpha: float = 0.05, thresholds: tuple[float, ...] = Z_THRESHOLDS) -> dict:
    """Compare observed upper-tail shares with their null (N1/N2) distributions; add-one Monte Carlo p."""
    out = {"test": "T4"}
    for t in thresholds:
        col = f"share_gt_{t:g}"
        r = placebo_p_value(observed[col], null_shares[col].to_numpy(), "greater")
        out[f"p_{col}"] = r["p"]
        out[f"reject_{col}"] = bool(r["p"] < alpha)
        out[f"null_q95_{col}"] = r["null_q95"]
    return out


def i2_one_sided_reject(tau2_draws: np.ndarray, alpha: float = 0.05, min_valid_share: float = 0.5) -> bool:
    """Bootstrap analogue of T2 (I2, I2b, I2c): reject tau2 = 0 when the alpha-quantile of the draws exceeds zero.

    Raises ValueError when fewer than ``min_valid_share`` of the draws are finite (a shrunken draw set is
    never scored silently); invalid draws are otherwise counted by the bootstrap itself."""
    d = np.asarray(tau2_draws, dtype=float)
    f = d[np.isfinite(d)]
    if len(d) == 0 or len(f) < min_valid_share * len(d):
        raise ValueError(f"only {len(f)} of {len(d)} bootstrap draws are finite")
    return bool(np.quantile(f, alpha) > 0.0)
