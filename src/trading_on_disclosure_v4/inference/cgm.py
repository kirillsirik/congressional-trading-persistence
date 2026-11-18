"""I1: two-way Cameron-Gelbach-Miller SE for the A1 WLS slope, wrapping the Act I implementation.

Wraps ``trading_on_disclosure_v2.inference.cgm3.cgm_cluster_cov`` (OLS-only) via the sqrt(w)
transform: WLS on (y, X, w) is OLS on (sqrt(w) y, sqrt(w) X), and the clustered sandwich of the
transformed rows is the WLS sandwich. Same per-term correction and dof = min cluster count - 1,
t-based p and CI with dof = max(dof, 1), exactly as ``locked_estimands.two_way_spread``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v2.inference.cgm3 import CGM3Result, cgm_cluster_cov

from ..estimators import tau as tau_mod

CLUSTER_COLS = ("household_id", "anchor_date")


@dataclass(frozen=True)
class ClusteredSlope:
    coef: float
    se: float
    t_stat: float
    p: float
    ci_low: float
    ci_high: float
    dof: int
    n_rows: int
    cluster_counts: dict[str, int]
    psd_corrected: bool
    level: float = 0.95


def wls_cluster_cov(
    y: np.ndarray, X: np.ndarray, w: np.ndarray, clusters: pd.DataFrame, small_sample: bool = True
) -> CGM3Result:
    """WLS coefficients with CGM multi-way covariance; ``clusters`` has 1-3 cluster columns."""
    sw = np.sqrt(np.asarray(w, dtype=float))
    if np.any(~np.isfinite(sw)) or np.any(sw <= 0):
        raise ValueError("weights must be finite and strictly positive")
    return cgm_cluster_cov(
        np.asarray(y, dtype=float) * sw,
        np.asarray(X, dtype=float) * sw[:, None],
        clusters.reset_index(drop=True),
        small_sample=small_sample,
    )


def clustered_slope(
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    clusters: pd.DataFrame,
    coef_index: int = 1,
    level: float = 0.95,
) -> ClusteredSlope:
    """Two-way (or multi-way) clustered inference on one WLS coefficient, Act I conventions."""
    res = wls_cluster_cov(y, X, w, clusters)
    est = float(res.beta[coef_index])
    se = float(np.sqrt(res.cov[coef_index, coef_index]))
    dof = max(int(res.dof), 1)
    t_stat = est / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan
    crit = float(stats.t.ppf(0.5 + level / 2, df=dof))
    return ClusteredSlope(
        coef=est,
        se=se,
        t_stat=float(t_stat) if np.isfinite(t_stat) else np.nan,
        p=p,
        ci_low=est - crit * se,
        ci_high=est + crit * se,
        dof=dof,
        n_rows=int(len(y)),
        cluster_counts=dict(res.cluster_counts),
        psd_corrected=bool(res.psd_corrected),
        level=level,
    )


def tau2_i1(
    panel: pd.DataFrame,
    weights: str = "ES",
    y: str = "y_h20",
    household: str = "household_id",
    period: str = "period",
    anchor: str = "anchor_date",
    activity_control: bool = True,
    level: float = 0.95,
) -> dict:
    """A1 with I1 inference: tau2 = b*Var_w(x); CI(tau2) = Var_w(x) x CI(b); one-sided upper bound."""
    table = tau_mod.household_table(panel, weights, y=y, household=household, period=period)
    ev = tau_mod.event_design(panel, table, y=y, household=household, period=period, anchor=anchor)
    if anchor not in ev.columns:
        raise ValueError(f"I1 needs the {anchor!r} column for two-way clustering")
    X = tau_mod.design_matrix(ev, activity_control)
    clusters = pd.DataFrame({CLUSTER_COLS[0]: ev[household].to_numpy(), CLUSTER_COLS[1]: ev[anchor].to_numpy()})
    sl = clustered_slope(ev["y"].to_numpy(), X, ev["event_weight"].to_numpy(), clusters, level=level)
    v = tau_mod.var_w(table.x, table.w)
    tau2 = sl.coef * v
    out = {
        "weights": weights, "activity_control": activity_control, "inference": "I1_cgm_two_way",
        "b_hat": sl.coef, "se_b": sl.se, "t_stat": sl.t_stat, "p_two_sided": sl.p, "dof": sl.dof,
        "var_wx": v, "tau2_hat": tau2, "tau_hat": tau_mod.signed_root(tau2),
        "n_households": int(len(table.household_id)), "n_events_a": int(table.n_a.sum()),
        "n_events_b": int(table.n_b.sum()), "n_anchor_dates": sl.cluster_counts.get(CLUSTER_COLS[1], 0),
        "psd_corrected": sl.psd_corrected, "n_dropped_nonfinite": table.n_dropped_nonfinite,
    }
    out.update(tau_mod.tau_ci_from_b_ci(sl.ci_low, sl.ci_high, v))
    out.update(tau_mod.tau_upper_bound(sl.coef, sl.se, v, dof=sl.dof, level=level))
    out["p_one_sided_tau2_positive"] = float(stats.t.sf(sl.t_stat, df=sl.dof)) if np.isfinite(sl.t_stat) else np.nan
    return out
